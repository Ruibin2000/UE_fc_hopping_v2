from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple, List
import os, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import trange, tqdm

THIS_DIR = Path(__file__).resolve().parent

mobility = "10mps"

# ----------------- User config -----------------
USER_CFG = dict(
    epochs=200,
    W=20,
    H=1,
    batch_size=512,
    lr=1e-3,
    weight_decay=0.0,
    hidden=128,         # GRU hidden size
    num_layers=1,
    dropout=0.2,
    seed=2025,

    data_dir=str(THIS_DIR / "dataset" / "newBandit_npz_0.2_150_") +mobility,
    pattern_prefix="bandit_1_",
    pattern_suffix="_" +mobility+ ".npz",
    test_file_path="test_0.2_150_" +mobility+ ".txt",

    cap_vec=(960.0, 960.0, 480.0, 480.0),  # or None to estimate from train files

    use_softcap=False,
    softcap_sharp=1.0,

    num_workers=4,
    ckpt=str(THIS_DIR / "newBandit_multi_0.2_150_") +mobility+ "_GRU_mSe_lr1e-3_W20_batch_512.pt",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
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
    p.add_argument("--num_layers", type=int)
    p.add_argument("--dropout", type=float)
    p.add_argument("--seed", type=int)

    p.add_argument("--data_dir", type=str)
    p.add_argument("--pattern_prefix", type=str)
    p.add_argument("--pattern_suffix", type=str)
    p.add_argument("--test_file_path", type=str)

    p.add_argument("--cap_vec", type=str, help="e.g. 960,960,480,480")
    p.add_argument("--use_softcap", type=int, choices=[0, 1])
    p.add_argument("--softcap_sharp", type=float)

    p.add_argument("--ckpt", type=str)
    p.add_argument("--device", type=str)  # allow cuda:0 / cuda:2 / cpu
    p.add_argument("--num_workers", type=int)

    args = vars(p.parse_args())
    args = {k: v for k, v in args.items() if v is not None}
    if "cap_vec" in args:
        s = args["cap_vec"]
        args["cap_vec"] = tuple(float(x) for x in s.split(",") if x.strip() != "")
    if "use_softcap" in args:
        args["use_softcap"] = bool(args["use_softcap"])
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
    return: X (W,N,3), R_tp1 (N,)
    """
    def __init__(self, npz_path: str, W: int, H: int = 1, cap_vec: Optional[Tuple[float, ...]] = None):
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
        X = build_history_features(RM_hist, SM_hist, M_hist)  # (W,N,3)
        X = torch.nan_to_num(X); R_tp1 = torch.nan_to_num(R_tp1)
        return X, R_tp1

# ----------------- GRU model (shared across antennas) -----------------
class RateEstimatorGRU(nn.Module):
    """
    Input:  X_hist (B, W, N, F)  [F=3]
    Shared GRU over time for each antenna:
      reshape -> (B*N, W, F) -> GRU -> last hidden -> head -> (B*N,1) -> reshape (B,N)
    """
    def __init__(self, num_antennas: int, feat_dim: int = 3,
                 hidden: int = 128, num_layers: int = 1, dropout: float = 0.2,
                 use_cap: bool = True, use_softcap: bool = False, softcap_sharp: float = 1.0):
        super().__init__()
        self.N = int(num_antennas)
        self.F = int(feat_dim)
        self.hidden = int(hidden)
        self.num_layers = int(num_layers)

        self.use_cap = bool(use_cap)
        self.use_softcap = bool(use_softcap)
        self.softcap_sharp = float(softcap_sharp)
        self.register_buffer("cap_vec", torch.ones(self.N), persistent=False)

        # NOTE: GRU dropout only works when num_layers > 1
        gru_drop = float(dropout) if self.num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=self.F,
            hidden_size=self.hidden,
            num_layers=self.num_layers,
            dropout=gru_drop,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden, self.hidden),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden, 1),
            nn.Softplus(beta=1.5),
        )

    def _apply_cap(self, y: torch.Tensor) -> torch.Tensor:
        if not self.use_cap or self.cap_vec is None:
            return y
        if self.use_softcap:
            denom = self.cap_vec[None, :] * self.softcap_sharp + 1e-6
            return self.cap_vec[None, :] * torch.sigmoid(y / denom)
        return torch.minimum(y, self.cap_vec[None, :])

    def forward(self, X_hist: torch.Tensor):
        X_hist = torch.nan_to_num(X_hist)
        assert X_hist.dim() == 4, f"X_hist shape={X_hist.shape}"
        B, W, N, F = X_hist.shape
        assert N == self.N and F == self.F, f"Expect N={self.N},F={self.F}, got N={N},F={F}"

        x = X_hist.permute(0, 2, 1, 3).contiguous()      # (B,N,W,F)
        x = x.view(B * N, W, F)                          # (B*N,W,F)
        out, h = self.gru(x)                             # h: (L,B*N,H)
        last = h[-1]                                     # (B*N,H)
        y = self.head(last).squeeze(-1)                  # (B*N,)
        y = y.view(B, N)                                 # (B,N)
        y = self._apply_cap(torch.nan_to_num(y))
        return {"R_hat": y}

# ----------------- evaluate on all antennas MSE -----------------
@torch.no_grad()
def eval_mse(model: nn.Module, loader: DataLoader, device: str) -> float:
    if loader is None: return float("inf")
    model.eval()
    tot, n = 0.0, 0
    for X, R in loader:
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
    for X, R in loader:
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
    test_file_path: str = "test_0.2_150_10mps.txt"

    W: int = 20
    H: int = 1
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 200
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt: str = str(THIS_DIR / "rate_estimator_allmse_GRU.pt")
    seed: int = 2025

    hidden: int = 128
    num_layers: int = 1
    dropout: float = 0.2

    cap_vec: Optional[Tuple[float, ...]] = None
    num_workers: int = 4

    use_softcap: bool = False
    softcap_sharp: float = 1.0

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
        r = np.load(p)["rates"].astype(np.float32)
        maxes.append(r.max(axis=0))
    cap = np.max(np.stack(maxes, 0), axis=0).astype(np.float32)
    cap = np.maximum(cap, 1.0)
    return cap

# ----------------- save ckpt (same format as your TFMR script) -----------------
def save_ckpt(path: str, model: nn.Module, opt: torch.optim.Optimizer, epoch: int,
              best_val: float, ds_N: int, feat_dim: int, cap_vec: np.ndarray, cfg_dict: dict):
    payload = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "epoch": epoch,
        "best_loss": float(best_val),
        "N": int(ds_N),
        "feat_dim": int(feat_dim),
        "z_dim": int(cfg_dict.get("hidden", 128)),  # keep key for compatibility (store hidden here)
        "cap_vec": cap_vec.astype(np.float32),
        "cfg": cfg_dict,
    }
    payload["hidden"] = int(cfg_dict.get("hidden", 128))
    payload["num_layers"] = int(cfg_dict.get("num_layers", 1))
    torch.save(payload, path)

def apply_overrides(cfg: TrainCfg, overrides: dict) -> TrainCfg:
    fields = {k: v for k, v in overrides.items() if hasattr(cfg, k)}
    return replace(cfg, **fields)

# ----------------- main -----------------
def main():
    cfg = TrainCfg()
    cfg = apply_overrides(cfg, USER_CFG)
    cfg = apply_overrides(cfg, parse_args())

    print("[DEVICE] cfg.device =", cfg.device)
    print("[DEVICE] torch.cuda.is_available =", torch.cuda.is_available())
    print("[DEVICE] torch.cuda.device_count =", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("[DEVICE] CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
        cur = torch.cuda.current_device()
        print("[DEVICE] current_device =", cur)
        print("[DEVICE] device_name =", torch.cuda.get_device_name(cur))

    set_seed(cfg.seed)

    files = list_bandit_files(cfg.data_dir, cfg.pattern_prefix, cfg.pattern_suffix)
    if len(files) == 0:
        raise FileNotFoundError(f"unable to find {cfg.pattern_prefix}*{cfg.pattern_suffix} at {cfg.data_dir}")

    train_idx, val_idx, test_idx = split_by_files(files, cfg.seed, ratios=(0.7, 0.15, 0.15))
    train_files = [files[i] for i in train_idx]
    val_files   = [files[i] for i in val_idx]
    test_files  = [files[i] for i in test_idx]

    # write test list
    with open(THIS_DIR / cfg.test_file_path, "w") as f:
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
    model = RateEstimatorGRU(
        num_antennas=N, feat_dim=3,
        hidden=cfg.hidden, num_layers=cfg.num_layers, dropout=cfg.dropout,
        use_cap=True, use_softcap=cfg.use_softcap, softcap_sharp=cfg.softcap_sharp,
    ).to(cfg.device)

    with torch.no_grad():
        model.cap_vec.copy_(torch.tensor(cap_vec, device=cfg.device))

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf"); saved_once = False
    last_val_mse = None
    first_debug_printed = False

    for epoch in trange(1, cfg.epochs + 1, desc="Epochs"):
        model.train()
        tot, n = 0.0, 0

        for X, R in train_loader:
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