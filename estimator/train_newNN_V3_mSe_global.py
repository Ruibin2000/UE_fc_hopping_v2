# ===================== train_estimator_multi_npz_all_mse.py Transformer =====================
from __future__ import annotations
from dataclasses import dataclass, replace
from typing import Optional, Tuple, List
from pathlib import Path
import random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import trange, tqdm

THIS_DIR = Path(__file__).resolve().parent

# ----------------- User config -----------------
USER_CFG = dict(
    # 训练超参
    epochs=200,
    W=20,
    H=1,
    batch_size=128,
    lr=1e-3,                
    weight_decay=0.0,
    hidden=32,               
    z_dim=128,            
    dropout=0.2,
    seed=2025,


    data_dir=str(THIS_DIR / "dataset" / "newBandit_npz_0.2_150_10mps"),
    pattern_prefix="bandit_1_",
    pattern_suffix="_10mps.npz",
    test_file_path="test_0.2_150_10mps.txt",

    # cap
    cap_vec=(960.0, 960.0, 480.0, 480.0),


    use_global_ctx=True,     
    use_softcap=False,
    softcap_sharp=1.0,


    num_heads=4,
    attn_mode="global",     # 'grouped' | 'global' | 'off'
    attn_bias=0.5,           # under grouped add off-diag
    use_ant_id=True,         # whether antenna ID embedding
    ffn_mult=4,              # FFN extend factor

    num_workers=2,
    ckpt=str(THIS_DIR / "newBandit_multi_0.2_150_10mps_mSe_lr1e-3_W20_V2_tfmr_global.pt"),
    device=f"cuda:{2}" if torch.cuda.is_available() else "cpu",
)

# ----------------- command line -----------------
def parse_args() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int)
    p.add_argument("--W", type=int)
    p.add_argument("--H", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--hidden", type=int)
    p.add_argument("--z_dim", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--data_dir", type=str)
    p.add_argument("--pattern_prefix", type=str)
    p.add_argument("--pattern_suffix", type=str)
    p.add_argument("--ckpt", type=str)
    p.add_argument("--device", type=str, choices=["cuda","cpu"])
    p.add_argument("--num_workers", type=int)
    p.add_argument("--cap_vec", type=str, help="e.g. 960,960,480,480")
    p.add_argument("--use_global_ctx", type=int, choices=[0,1])  
    p.add_argument("--use_softcap", type=int, choices=[0,1])
    p.add_argument("--softcap_sharp", type=float)
    p.add_argument("--num_heads", type=int)
    # Transformer Config
    p.add_argument("--attn_mode", type=str, choices=["grouped","global","off"])
    p.add_argument("--attn_bias", type=float)
    p.add_argument("--use_ant_id", type=int, choices=[0,1])
    p.add_argument("--ffn_mult", type=int)
    args = vars(p.parse_args())
    args = {k:v for k,v in args.items() if v is not None}
    if "cap_vec" in args:
        s = args["cap_vec"]
        args["cap_vec"] = tuple(float(x) for x in s.split(",") if x.strip()!="")
    if "use_global_ctx" in args:
        args["use_global_ctx"] = bool(args["use_global_ctx"])
    if "use_softcap" in args:
        args["use_softcap"] = bool(args["use_softcap"])
    if "use_ant_id" in args:
        args["use_ant_id"] = bool(args["use_ant_id"])
    return args

# ----------------- tools -----------------
def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_history_features(RM_hist: torch.Tensor, SM_hist: torch.Tensor, M_hist: torch.Tensor) -> torch.Tensor:
    return torch.stack([RM_hist, SM_hist, M_hist], dim=-1)  # (W,N,3)

# ----------------- dataset -----------------
class MasksOnlyDatasetNext(Dataset):
    """
    .npz: rate_mask, snr_mask, mask, rates shape (T,N)
    """
    def __init__(self, npz_path: str, W: int, H: int = 1,
                 cap_vec: Optional[Tuple[float, ...]] = None):
        super().__init__()
        self.path = str(npz_path)
        data = np.load(npz_path)
        rates = data["rates"].astype(np.float32)      
        rm    = data["rate_mask"].astype(np.float32)  
        sm    = data["snr_mask"].astype(np.float32)
        m     = data["mask"].astype(np.float32)

        T, N = rates.shape
        if cap_vec is not None:
            cap = np.asarray(cap_vec, dtype=np.float32); assert cap.shape == (N,)
            rates = np.clip(rates, 0.0, cap[None, :])
            rm    = np.clip(rm,    0.0, cap[None, :])
            self.cap_vec = cap
        else:
            self.cap_vec = None

        self.rm, self.sm, self.mask, self.rates = rm, sm, m, rates
        self.T, self.N = T, N
        self.W, self.H = int(W), int(H)
        if self.T <= self.W + self.H:
            raise ValueError(f"T must > W+H({self.path})")

    def __len__(self): return self.T - self.W - self.H + 1

    def __getitem__(self, idx: int):
        s, e = idx, idx + self.W
        t = e - 1
        tp1 = t + self.H
        RM_hist = torch.tensor(self.rm[s:e],   dtype=torch.float32)
        SM_hist = torch.tensor(self.sm[s:e],   dtype=torch.float32)
        M_hist  = torch.tensor(self.mask[s:e], dtype=torch.float32)
        R_tp1   = torch.tensor(self.rates[tp1], dtype=torch.float32)  
        A_t     = torch.tensor(self.mask[t],    dtype=torch.float32)
        X = build_history_features(RM_hist, SM_hist, M_hist)          # (W,N,3)

        X = torch.nan_to_num(X); R_tp1 = torch.nan_to_num(R_tp1); A_t = torch.nan_to_num(A_t)
        return X, R_tp1, A_t

# ----------------- modular-----------------
class LearnableAffine(nn.Module):
    """
    x' = alpha ⊙ x + beta  
    """
    def __init__(self, feat_dim: int, min_alpha: float = 1e-3):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(feat_dim))
        self.beta  = nn.Parameter(torch.zeros(feat_dim))
        self.min_alpha = float(min_alpha)

    def forward(self, x):  # x: (..., F)
        alpha = self.alpha.clamp_min(self.min_alpha)
        return x * alpha + self.beta

# ----------------- light time Conv encoder -----------------
class TemporalConvEncoder(nn.Module):
    """
    输入:  (B, N, W, C)  [C = 2F: raw + scaled]
    输出:  (B, N, D)     [per-antenna embedding]
    Pipeline: DWConv(k=5) -> PWConv(C->D) -> LN -> PWConv(D->D) -> AvgPool(W)
    """
    def __init__(self, in_channels: int, hidden: int = 64, out_dim: int = 64, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=pad, groups=in_channels)
        self.pw1 = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.ln  = nn.LayerNorm(hidden)  # applied on (B*N, W, hidden)
        self.pw2 = nn.Conv1d(hidden, out_dim, kernel_size=1)

    def forward(self, x):  # x: (B, N, W, C)
        B, N, W, C = x.shape
        y = x.reshape(B * N, W, C).transpose(1, 2)  # (B*N, C, W)
        y = self.dw(y)
        y = self.pw1(self.act(y))                   # (B*N, hidden, W)
        y = y.transpose(1, 2)                       # (B*N, W, hidden)
        y = self.ln(y)
        y = y.transpose(1, 2)                       # (B*N, hidden, W)
        y = self.pw2(self.act(y))                   # (B*N, D, W)
        z = y.mean(dim=-1).view(B, N, -1)           # (B, N, D)
        return z

# ----------------- single layer Transformer -----------------
class AntennaTransformer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1,
                 attn_mode: str = "grouped", groups=None, attn_bias: float = 0.5,
                 use_ant_id: bool = True, ffn_mult: int = 4):
        super().__init__()
        assert attn_mode in ("grouped", "global", "off")
        self.dim = int(dim)
        self.mode = attn_mode
        self.groups = groups if groups is not None else []
        self.attn_bias_val = float(attn_bias)
        self.use_ant_id = bool(use_ant_id)

        if self.mode == "off":
            self.ln_in  = nn.Identity()
            self.mha    = None
            self.ffn_ln = nn.Identity()
            self.ffn    = nn.Identity()
            self.drop   = nn.Dropout(dropout)
            self.ant_emb = None
            return

        def resolve_heads(h, d):
            if d % h == 0:
                return h
            for k in range(min(h, d), 0, -1):
                if d % k == 0:
                    return k
            return 1
        self.num_heads = resolve_heads(num_heads, self.dim)

        self.ln_in = nn.LayerNorm(self.dim)
        self.ant_emb = nn.Parameter(torch.zeros(1, 1, self.dim)) if self.use_ant_id else None
        if self.use_ant_id:
            nn.init.normal_(self.ant_emb, std=0.02)

        self.mha = nn.MultiheadAttention(embed_dim=self.dim,
                                         num_heads=self.num_heads,
                                         dropout=dropout,
                                         batch_first=True)
        self.drop = nn.Dropout(dropout)

        hidden = self.dim * int(ffn_mult)
        self.ffn_ln = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.dim),
            nn.Dropout(dropout),
        )

    @staticmethod
    def _complete_groups(N: int, groups):
        if groups is None:
            return [[i] for i in range(N)]
        in_any = set(i for g in groups for i in g)
        full = [list(g) for g in groups] + [[i] for i in range(N) if i not in in_any]
        return full

    @staticmethod
    def _build_group_mask_and_bias(N: int, groups, bias_val: float, device, dtype):

        full = AntennaTransformer._complete_groups(N, groups)
        allow = torch.zeros((N, N), device=device, dtype=dtype)
        for g in full:
            idx = torch.tensor(g, device=device, dtype=torch.long)
            allow[idx.unsqueeze(1), idx.unsqueeze(0)] = 1.0
        neg = torch.tensor(-1e4, device=device, dtype=dtype)
        mask = torch.where(allow > 0, torch.zeros_like(allow), neg)
        if bias_val != 0.0:
            bias = torch.zeros_like(allow)
            for g in full:
                idx = torch.tensor(g, device=device, dtype=torch.long)
                if len(idx) > 1:
                    bias[idx.unsqueeze(1), idx.unsqueeze(0)] = bias_val
                    bias[range(N), range(N)] = 0.0
            mask = mask + bias
        return mask  # (N, N)

    def forward(self, z: torch.Tensor, groups=None):
        """
        z: (B, N, D)
        """
        if self.mode == "off" or self.mha is None:
            return z

        B, N, D = z.shape
        x = self.ln_in(z)

        if self.ant_emb is not None:
            x = x + self.ant_emb.expand(B, N, D)

        if self.mode == "global":
            attn_mask = None
        else:
            g = groups if groups is not None else self.groups
            attn_mask = self._build_group_mask_and_bias(N, g, self.attn_bias_val, z.device, z.dtype)

        y, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)  # (B,N,D)
        z = z + self.drop(y)
        y2 = self.ffn(self.ffn_ln(z))
        z = z + y2
        return z

# ----------------- RateEstimatorV3 -----------------
class RateEstimatorV2(nn.Module):
    """
    Pipeline (B, W, N, F):
      1) LearnableAffine scaling (reversible) (B,W,N,2F)
      2) TemporalConvEncoder over time -> (B,N,z_dim)   [inter (B,N,W,C)]
      3) AntennaTransformer over antennas -> (B,N,z_dim)  [mode: grouped/global/off]
      4) Append per-antenna raw mean over time -> (B,N,z_dim+F)
      5) Shared MLP + Softplus -> (B,N) finally hard cap
    """
    def __init__(self, num_antennas: int, feat_dim: int = 3, z_dim: int = 64,
                 dropout: float = 0.1,
                 use_cap: bool = True,
                 use_softcap: bool = False,
                 softcap_sharp: float = 1.0,
                 groups=None,
                 num_heads: int = 4,
                 attn_mode: str = "grouped",     # 'grouped' | 'global' | 'off'
                 attn_bias: float = 0.5,
                 use_ant_id: bool = True,
                 ffn_mult: int = 4):
        super().__init__()
        self.N = int(num_antennas)
        self.F = int(feat_dim)
        self.z_dim = int(z_dim)
        self.use_cap = bool(use_cap)
        self.use_softcap = bool(use_softcap)
        self.softcap_sharp = float(softcap_sharp)

        if groups is None:
            groups = [[0, 1], [2, 3]] if self.N >= 4 else [list(range(self.N))]
        self.groups = groups

        self.register_buffer("cap_vec", torch.ones(self.N), persistent=False)

        self.scaler = LearnableAffine(self.F)
        in_C = self.F * 2

        self.temporal = TemporalConvEncoder(in_channels=in_C, hidden=z_dim, out_dim=z_dim)

        self.interact = AntennaTransformer(
            dim=z_dim, num_heads=num_heads, dropout=dropout,
            attn_mode=attn_mode, groups=self.groups, attn_bias=attn_bias,
            use_ant_id=use_ant_id, ffn_mult=ffn_mult
        )

        head_in = z_dim + self.F
        self.head = nn.Sequential(
            nn.Linear(head_in, z_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(z_dim, 1),
            nn.Softplus(beta=1.5)  # ensure non-negative
        )

    def _apply_cap(self, y: torch.Tensor) -> torch.Tensor:
        if not self.use_cap or self.cap_vec is None:
            return y
        if self.use_softcap:
            denom = self.cap_vec[None, :] * self.softcap_sharp + 1e-6
            return self.cap_vec[None, :] * torch.sigmoid(y / denom)
        else:
            return torch.minimum(y, self.cap_vec[None, :])

    def forward(self, X_hist):  # (B, W, N, F)
        assert X_hist.dim() == 4, f"X_hist shape={X_hist.shape}"
        X_hist = torch.nan_to_num(X_hist)

        B, W, N, F = X_hist.shape
        assert N == self.N and F == self.F, f"Expect N={self.N},F={self.F}, got N={N},F={F}"

        X_scaled = self.scaler(X_hist)                  # (B,W,N,F)
        x = torch.cat([X_hist, X_scaled], dim=-1)       # (B,W,N,2F)

        # time encoder process as (B,N,W,C)
        x_perm = x.permute(0, 2, 1, 3).contiguous()     # (B,N,W,2F)
        z = self.temporal(x_perm)                       # (B,N,z_dim)
        z = torch.nan_to_num(z)

        z = self.interact(z, groups=self.groups)        # (B,N,z_dim)
        z = torch.nan_to_num(z)

        raw_mean = torch.nan_to_num(X_hist.mean(dim=1)) # (B,N,F)
        h = torch.cat([z, raw_mean], dim=-1)            # (B,N,z_dim+F)

        y = self.head(h).squeeze(-1)                    # (B,N)
        y = torch.nan_to_num(y)
        y = self._apply_cap(y)
        return {"R_hat": y}

# ----------------- evaluate on all antennas MSE -----------------
@torch.no_grad()
def eval_mse(model: nn.Module, loader: DataLoader, device: str) -> float:
    if loader is None: return float("inf")
    model.eval()
    tot, n = 0.0, 0
    for X, R, _ in loader:
        X = X.to(device, non_blocking=True); R = R.to(device, non_blocking=True)
        out = model(X)["R_hat"]
        loss = F.mse_loss(out, R, reduction="mean")
        b = R.size(0); tot += loss.item() * b; n += b
    return tot / max(1, n)

@torch.no_grad()
def per_antenna_mse(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:

    if loader is None:
        return np.zeros(1, dtype=np.float32)
    model.eval()
    tot = None; cnt = 0
    for X, R, _ in loader:
        X = X.to(device, non_blocking=True); R = R.to(device, non_blocking=True)
        P = model(X)["R_hat"]
        e2 = (P - R).pow(2)  # (B,N)
        s = e2.sum(dim=0).detach().cpu().numpy()
        tot = s if tot is None else (tot + s)
        cnt += R.size(0)
    return (tot / max(1, cnt)) if tot is not None else np.zeros(1, dtype=np.float32)

# ----------------- dataclass -----------------
@dataclass
class TrainCfg:
    data_dir: str = str(THIS_DIR / "npz")
    pattern_prefix: str = "bandit_1_"
    pattern_suffix: str = "_.npz"
    test_file_path: str = "test_0.2_150_10mps"
    W: int = 200
    H: int = 1
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 300
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt: str = str(THIS_DIR / "rate_estimator_allmse_V2_tfmr_fp32.pt")
    seed: int = 2025
    hidden: int = 128          # 兼容字段（V2 忽略）
    z_dim: int = 128
    dropout: float = 0.2
    cap_vec: Optional[Tuple[float, ...]] = None
    num_workers: int = 2
    # 兼容/策略字段
    use_global_ctx: bool = True     # V2 忽略
    use_softcap: bool = False
    softcap_sharp: float = 1.0
    # Transformer 相关
    num_heads: int = 4
    attn_mode: str = "grouped"
    attn_bias: float = 0.5
    use_ant_id: bool = True
    ffn_mult: int = 4

# ----------------- list & split files -----------------
def list_bandit_files(data_dir: str, pre: str, suf: str) -> List[Path]:
    d = Path(data_dir)
    return sorted([p for p in d.glob(f"{pre}*{suf}") if p.is_file()])

def split_by_files(files: List[Path], seed: int, ratios=(0.7, 0.15, 0.15)):
    assert abs(sum(ratios) - 1.0) < 1e-6
    rnd = random.Random(seed)
    idx = list(range(len(files))); rnd.shuffle(idx)
    n = len(files)
    n_train = int(round(ratios[0] * n))
    n_val   = int(round(ratios[1] * n))
    n_train = min(n_train, n - 2)
    n_val   = min(n_val,   n - n_train - 1)
    n_test  = n - n_train - n_val
    return idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

def estimate_cap_vec_by_max(train_files: List[Path]) -> np.ndarray:

    assert len(train_files) > 0
    N = np.load(train_files[0])["rates"].shape[1]
    maxes = []
    for p in train_files:
        r = np.load(p)["rates"].astype(np.float32)   # (T,N)
        maxes.append(r.max(axis=0))                  # (N,)
    cap = np.max(np.stack(maxes, 0), axis=0).astype(np.float32)
    cap = np.maximum(cap, 1.0)
    return cap

# ----------------- save ckpt-----------------
def save_ckpt(path: str, model: nn.Module, opt: torch.optim.Optimizer, epoch: int,
              best_val: float, ds_N: int, feat_dim: int, cap_vec: np.ndarray, cfg_dict: dict):
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": epoch,
        "best_loss": float(best_val),
        "N": int(ds_N),
        "feat_dim": int(feat_dim),
        "z_dim": int(getattr(model, "z_dim", cfg_dict.get("z_dim", 128))),
        "cap_vec": cap_vec.astype(np.float32),
        "cfg": cfg_dict,
    }
    if hasattr(model, "hidden"):
        payload["hidden"] = int(getattr(model, "hidden"))
    torch.save(payload, path)


def apply_overrides(cfg: TrainCfg, overrides: dict) -> TrainCfg:
    fields = {k:v for k,v in overrides.items() if hasattr(cfg, k)}
    return replace(cfg, **fields)

# ----------------- main train -----------------
def main():

    cfg = TrainCfg()
    cfg = apply_overrides(cfg, USER_CFG)

    set_seed(cfg.seed)

    files = list_bandit_files(cfg.data_dir, cfg.pattern_prefix, cfg.pattern_suffix)
    if len(files) == 0:
        raise FileNotFoundError(f"unable to find {cfg.pattern_prefix}*{cfg.pattern_suffix} at {cfg.data_dir}")
    train_idx, val_idx, test_idx = split_by_files(files, cfg.seed, ratios=(0.7, 0.15, 0.15))
    train_files = [files[i] for i in train_idx]
    val_files   = [files[i] for i in val_idx]
    test_files  = [files[i] for i in test_idx]

    test_files_path = cfg.test_file_path
    with open(THIS_DIR / test_files_path, "w") as f:
        for p in test_files:
            f.write(str(p) + "\n")

    tqdm.write(f"[Split-by-file] total={len(files)} -> train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")


    if cfg.cap_vec is not None:
        cap_vec = np.asarray(cfg.cap_vec, dtype=np.float32)
    else:
        cap_vec = estimate_cap_vec_by_max(train_files)


    def make_concat(file_list: List[Path]) -> ConcatDataset:
        dss = []
        for p in file_list:
            ds = MasksOnlyDatasetNext(str(p), W=cfg.W, H=cfg.H, cap_vec=tuple(cap_vec))
            dss.append(ds)
        return ConcatDataset(dss)

    train_ds = make_concat(train_files)
    val_ds   = make_concat(val_files) if len(val_files) else None

    def make_loader(ds, shuffle: bool):
        if ds is None: return None

        kwargs = dict(
            batch_size=cfg.batch_size, shuffle=shuffle, drop_last=False,
            num_workers=cfg.num_workers, pin_memory=True,
            persistent_workers=(cfg.num_workers > 0),
        )
        if cfg.num_workers > 0:
            kwargs["prefetch_factor"] = 4
        return DataLoader(ds, **kwargs)

    train_loader = make_loader(train_ds, True)
    val_loader   = make_loader(val_ds, False)

    N = np.load(train_files[0])["rates"].shape[1]
    model = RateEstimatorV2(
        num_antennas=N, feat_dim=3, z_dim=cfg.z_dim, dropout=cfg.dropout,
        use_cap=True, use_softcap=cfg.use_softcap, softcap_sharp=cfg.softcap_sharp,
        groups=None, num_heads=cfg.num_heads,
        attn_mode=cfg.attn_mode, attn_bias=cfg.attn_bias,
        use_ant_id=cfg.use_ant_id, ffn_mult=cfg.ffn_mult
    ).to(cfg.device)


    with torch.no_grad():
        model.cap_vec.copy_(torch.tensor(cap_vec, device=cfg.device))

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf"); saved_once = False
    last_val_mse = None

    first_debug_printed = False

    for epoch in trange(1, cfg.epochs+1, desc="Epochs"):
        model.train()
        tot, n = 0.0, 0
        for X, R, _A in train_loader:
            X = X.to(cfg.device, non_blocking=True)
            R = R.to(cfg.device, non_blocking=True)

            if not first_debug_printed:
                with torch.no_grad():
                    print("[DEBUG] X finite:", torch.isfinite(X).all().item(),
                          "min/max:", float(X.min()), float(X.max()),
                          "shape:", tuple(X.shape))
                first_debug_printed = True

            opt.zero_grad(set_to_none=True)

            out = model(X)["R_hat"]
            loss = F.mse_loss(out, R, reduction="mean")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            b = R.size(0); tot += loss.item() * b; n += b
        train_mse = tot / max(1, n)

        last_val_mse = eval_mse(model, val_loader, cfg.device) if val_loader is not None else train_mse

        if val_loader is not None:
            mse_vec = per_antenna_mse(model, val_loader, cfg.device)
            mse_str = " ".join(f"{v:.6f}" for v in mse_vec)
            tqdm.write(f"[Epoch {epoch:03d}] per-antenna val MSE: {mse_str}")

        tqdm.write(f"[Epoch {epoch:03d}] train_mse(all)={train_mse:.6f}  val_mse(all)={last_val_mse:.6f}")

        if last_val_mse < best_val - 1e-9:
            best_val = last_val_mse
            save_ckpt(cfg.ckpt, model, opt, epoch, best_val, ds_N=N, feat_dim=3,
                      cap_vec=cap_vec, cfg_dict=dict(vars(cfg)))
            saved_once = True
            tqdm.write(f"  ↳ saved (best val MSE={best_val:.6f}) -> {cfg.ckpt}")

    if not saved_once:
        fallback_val = last_val_mse if last_val_mse is not None else float("inf")
        save_ckpt(cfg.ckpt, model, opt, cfg.epochs, fallback_val, ds_N=N, feat_dim=3,
                  cap_vec=cap_vec, cfg_dict=dict(vars(cfg)))
        tqdm.write(f"  ↳ saved (last epoch fallback) -> {cfg.ckpt}")

    print(f"Finished. best_val_mse={best_val:.6f}. Test files saved to {cfg.test_file_path}")

if __name__ == "__main__":
    main()
