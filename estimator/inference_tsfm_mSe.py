from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

THIS_DIR = Path(__file__).resolve().parent

mobility = "10mps"
USER_CFG = dict(
    ckpt=str(THIS_DIR / "newBandit_multi_0.2_150_") +mobility+ str("_tsfm_mSe_lr1e-3_W20_batch_512.pt"),
    test_list=str(THIS_DIR / "test_0.2_150_") +mobility+ ".txt",
    out_dir=str(THIS_DIR / "pred_json_tfmr"),
    batch_size=512,
    device="cuda:0" if torch.cuda.is_available() else "cpu",  # NOTE: use cuda:0 form!
    W=None,
    H=None,
    first_k=None,
)

def parse_args() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str)
    p.add_argument("--test_list", type=str)
    p.add_argument("--out_dir", type=str)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--device", type=str)  # allow cuda:0, cuda:1, cpu
    p.add_argument("--W", type=int)
    p.add_argument("--H", type=int)
    p.add_argument("--first_k", type=int)
    args = vars(p.parse_args())
    return {k: v for k, v in args.items() if v is not None}

def build_history_features(RM_hist: torch.Tensor, SM_hist: torch.Tensor, M_hist: torch.Tensor) -> torch.Tensor:
    return torch.stack([RM_hist, SM_hist, M_hist], dim=-1)  # (W,N,3)

class MasksOnlyDatasetNext(Dataset):
    def __init__(self, npz_path: str, W: int, H: int, cap_vec: Optional[np.ndarray]):
        data = np.load(npz_path)
        rates = data["rates"].astype(np.float32)
        rm    = data["rate_mask"].astype(np.float32)
        sm    = data["snr_mask"].astype(np.float32)
        m     = data["mask"].astype(np.float32)

        if cap_vec is not None:
            cap_vec = cap_vec.astype(np.float32)
            rates = np.clip(rates, 0.0, cap_vec[None, :])
            rm    = np.clip(rm,    0.0, cap_vec[None, :])

        self.rm, self.sm, self.m, self.rates = rm, sm, m, rates
        self.T, self.N = rates.shape
        self.W, self.H = int(W), int(H)
        if self.T <= self.W + self.H:
            raise ValueError(f"T must > W+H for {npz_path}")

    def __len__(self):
        return self.T - self.W - self.H + 1

    def __getitem__(self, idx: int):
        s, e = idx, idx + self.W
        t = e - 1
        tp1 = t + self.H
        RM_hist = torch.tensor(self.rm[s:e], dtype=torch.float32)
        SM_hist = torch.tensor(self.sm[s:e], dtype=torch.float32)
        M_hist  = torch.tensor(self.m[s:e],  dtype=torch.float32)
        X = build_history_features(RM_hist, SM_hist, M_hist)  # (W,N,3)
        X = torch.nan_to_num(X)
        return X, int(tp1)

# ----------------- Model: exactly your TFMR -----------------
class LearnableAffine(nn.Module):
    def __init__(self, feat_dim: int, min_alpha: float = 1e-3):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(feat_dim))
        self.beta  = nn.Parameter(torch.zeros(feat_dim))
        self.min_alpha = float(min_alpha)
    def forward(self, x):
        alpha = self.alpha.clamp_min(self.min_alpha)
        return x * alpha + self.beta

class TemporalConvEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, out_dim: int = 64, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.dw  = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=pad, groups=in_channels)
        self.pw1 = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.ln  = nn.LayerNorm(hidden)
        self.pw2 = nn.Conv1d(hidden, out_dim, kernel_size=1)
    def forward(self, x):  # (B,N,W,C)
        B, N, W, C = x.shape
        y = x.reshape(B * N, W, C).transpose(1, 2)  # (B*N,C,W)
        y = self.dw(y)
        y = self.pw1(self.act(y))                   # (B*N,hidden,W)
        y = y.transpose(1, 2)                       # (B*N,W,hidden)
        y = self.ln(y)
        y = y.transpose(1, 2)                       # (B*N,hidden,W)
        y = self.pw2(self.act(y))                   # (B*N,D,W)
        z = y.mean(dim=-1).view(B, N, -1)
        return z

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
            if d % h == 0: return h
            for k in range(min(h, d), 0, -1):
                if d % k == 0: return k
            return 1
        self.num_heads = resolve_heads(num_heads, self.dim)

        self.ln_in = nn.LayerNorm(self.dim)
        self.ant_emb = nn.Parameter(torch.zeros(1, 1, self.dim)) if self.use_ant_id else None
        if self.use_ant_id:
            nn.init.normal_(self.ant_emb, std=0.02)

        self.mha = nn.MultiheadAttention(embed_dim=self.dim, num_heads=self.num_heads,
                                         dropout=dropout, batch_first=True)
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
        return mask

    def forward(self, z: torch.Tensor, groups=None):
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

        y, _ = self.mha(x, x, x, attn_mask=attn_mask, need_weights=False)
        z = z + self.drop(y)
        z = z + self.ffn(self.ffn_ln(z))
        return z

class RateEstimatorV2(nn.Module):
    def __init__(self, num_antennas: int, feat_dim: int = 3, z_dim: int = 64,
                 dropout: float = 0.1,
                 use_cap: bool = True,
                 use_softcap: bool = False,
                 softcap_sharp: float = 1.0,
                 groups=None,
                 num_heads: int = 4,
                 attn_mode: str = "grouped",
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
            nn.Softplus(beta=1.5),
        )

    def _apply_cap(self, y: torch.Tensor) -> torch.Tensor:
        if not self.use_cap or self.cap_vec is None:
            return y
        if self.use_softcap:
            denom = self.cap_vec[None, :] * self.softcap_sharp + 1e-6
            return self.cap_vec[None, :] * torch.sigmoid(y / denom)
        else:
            return torch.minimum(y, self.cap_vec[None, :])

    def forward(self, X_hist):
        X_hist = torch.nan_to_num(X_hist)
        B, W, N, F = X_hist.shape
        assert N == self.N and F == self.F

        X_scaled = self.scaler(X_hist)
        x = torch.cat([X_hist, X_scaled], dim=-1)         # (B,W,N,2F)
        x_perm = x.permute(0, 2, 1, 3).contiguous()       # (B,N,W,2F)
        z = self.temporal(x_perm)
        z = self.interact(z, groups=self.groups)
        raw_mean = torch.nan_to_num(X_hist.mean(dim=1))   # (B,N,F)
        h = torch.cat([z, raw_mean], dim=-1)
        y = self.head(h).squeeze(-1)
        y = self._apply_cap(torch.nan_to_num(y))
        return {"R_hat": y}

def load_ckpt_tfmr(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    N = int(ckpt["N"])
    F = int(ckpt.get("feat_dim", 3))
    z_dim = int(ckpt.get("z_dim", 128))
    cap_vec = np.asarray(ckpt["cap_vec"], dtype=np.float32)
    cfg = ckpt.get("cfg", {})

    dropout = float(cfg.get("dropout", 0.2))
    use_softcap = bool(cfg.get("use_softcap", False))
    softcap_sharp = float(cfg.get("softcap_sharp", 1.0))
    num_heads = int(cfg.get("num_heads", 4))
    attn_mode = str(cfg.get("attn_mode", "grouped"))
    attn_bias = float(cfg.get("attn_bias", 0.5))
    use_ant_id = bool(cfg.get("use_ant_id", True))
    ffn_mult = int(cfg.get("ffn_mult", 4))

    model = RateEstimatorV2(
        num_antennas=N, feat_dim=F, z_dim=z_dim, dropout=dropout,
        use_cap=True, use_softcap=use_softcap, softcap_sharp=softcap_sharp,
        groups=None, num_heads=num_heads,
        attn_mode=attn_mode, attn_bias=attn_bias,
        use_ant_id=use_ant_id, ffn_mult=ffn_mult
    ).to(device)

    sd = ckpt["model"]
    model_keys = set(model.state_dict().keys())
    trimmed = {k: v for k, v in sd.items() if k in model_keys}
    model.load_state_dict(trimmed, strict=False)

    with torch.no_grad():
        model.cap_vec.copy_(torch.tensor(cap_vec, device=device))
    model.eval()
    return model, cap_vec, cfg

@torch.no_grad()
def infer_file(npz_path: str, out_json: Path, model: nn.Module,
               cap_vec: np.ndarray, device: str,
               W: int, H: int, batch_size: int,
               first_k: Optional[int] = None):
    ds = MasksOnlyDatasetNext(npz_path, W=W, H=H, cap_vec=cap_vec)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    t_index: List[int] = []
    R_hat: List[List[float]] = []

    for X, tp1 in loader:
        X = X.to(device, non_blocking=True)
        pred = model(X)["R_hat"].detach().cpu().numpy()  # (B,N)
        for i in range(pred.shape[0]):
            t_index.append(int(tp1[i]))
            R_hat.append(pred[i].astype(np.float32).tolist())
            if first_k is not None and len(t_index) >= first_k:
                break
        if first_k is not None and len(t_index) >= first_k:
            break

    payload: Dict[str, Any] = dict(
        npz=str(Path(npz_path).resolve()),
        W=int(W), H=int(H),
        t_index=t_index,
        R_hat=R_hat,
        cap_vec=cap_vec.astype(np.float32).tolist(),
        model_type="transformer",
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f)
    print(f"[TFMR] saved -> {out_json}")

def main():
    cfg = USER_CFG.copy()
    cfg.update(parse_args())

    ckpt_path = Path(cfg["ckpt"])
    test_list = Path(cfg["test_list"])
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    assert ckpt_path.exists(), f"ckpt not found: {ckpt_path}"
    assert test_list.exists(), f"test list not found: {test_list}"

    device = cfg["device"]
    model, cap_vec, train_cfg = load_ckpt_tfmr(str(ckpt_path), device=device)

    W = int(cfg["W"]) if cfg.get("W") is not None else int(train_cfg.get("W", 20))
    H = int(cfg["H"]) if cfg.get("H") is not None else int(train_cfg.get("H", 1))
    bs = int(cfg["batch_size"])
    first_k = cfg.get("first_k")

    with open(test_list) as f:
        files = [Path(line.strip()) for line in f if line.strip()]

    for p in files:
        out_json = out_dir / f"{p.stem}_pred_tfmr.json"
        infer_file(str(p), out_json, model, cap_vec, device, W, H, bs, first_k=first_k)

if __name__ == "__main__":
    main()