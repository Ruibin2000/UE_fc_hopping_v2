from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any, List
import argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

THIS_DIR = Path(__file__).resolve().parent

mobility = "10mps"
USER_CFG = dict(
    ckpt=str(THIS_DIR / "newBandit_multi_0.2_150_") +mobility+ str("_GRU_mSe_lr1e-4_W20_batch_512.pt"),
    test_list=str(THIS_DIR / "test_0.2_150_") +mobility+ ".txt",
    out_dir=str(THIS_DIR / "pred_json_GRU_lr1e-4"),
    batch_size=512,
    device="cuda:0" if torch.cuda.is_available() else "cpu",
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
    p.add_argument("--device", type=str)
    p.add_argument("--W", type=int)
    p.add_argument("--H", type=int)
    p.add_argument("--first_k", type=int)
    args = vars(p.parse_args())
    return {k: v for k, v in args.items() if v is not None}

def torch_load_compat(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)  # PyTorch 2.6+
    except TypeError:
        return torch.load(path, map_location=map_location)  # older torch

def build_history_features(RM_hist: torch.Tensor, SM_hist: torch.Tensor, M_hist: torch.Tensor) -> torch.Tensor:
    return torch.stack([RM_hist, SM_hist, M_hist], dim=-1)

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

class RateEstimatorGRU(nn.Module):
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

        gru_drop = float(dropout) if self.num_layers > 1 else 0.0
        self.gru = nn.GRU(self.F, self.hidden, num_layers=self.num_layers,
                          dropout=gru_drop, batch_first=True, bidirectional=False)
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
        B, W, N, F = X_hist.shape
        x = X_hist.permute(0, 2, 1, 3).contiguous().view(B * N, W, F)
        _, h = self.gru(x)     # (L,B*N,H)
        last = h[-1]           # (B*N,H)
        y = self.head(last).squeeze(-1).view(B, N)
        y = self._apply_cap(torch.nan_to_num(y))
        return {"R_hat": y}

def load_ckpt_gru(ckpt_path: str, device: str):
    ckpt = torch_load_compat(ckpt_path, map_location="cpu")
    N = int(ckpt["N"])
    F = int(ckpt.get("feat_dim", 3))
    cap_vec = np.asarray(ckpt["cap_vec"], dtype=np.float32)
    cfg = ckpt.get("cfg", {})

    hidden = int(cfg.get("hidden", ckpt.get("hidden", 128)))
    num_layers = int(cfg.get("num_layers", ckpt.get("num_layers", 1)))
    dropout = float(cfg.get("dropout", 0.2))
    use_softcap = bool(cfg.get("use_softcap", False))
    softcap_sharp = float(cfg.get("softcap_sharp", 1.0))

    model = RateEstimatorGRU(
        num_antennas=N, feat_dim=F,
        hidden=hidden, num_layers=num_layers, dropout=dropout,
        use_cap=True, use_softcap=use_softcap, softcap_sharp=softcap_sharp
    ).to(device)

    sd = ckpt["model"]
    model.load_state_dict(sd, strict=False)
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
        model_type="gru",
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f)
    print(f"[GRU] saved -> {out_json}")

def main():
    cfg = USER_CFG.copy()
    cfg.update(parse_args())

    ckpt_path = Path(cfg["ckpt"])
    test_list = Path(cfg["test_list"])
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    assert ckpt_path.exists(), f"ckpt not found: {ckpt_path}"
    assert test_list.exists(), f"test list not found: {test_list}"

    device = cfg["device"]
    model, cap_vec, train_cfg = load_ckpt_gru(str(ckpt_path), device=device)

    W = int(cfg["W"]) if cfg.get("W") is not None else int(train_cfg.get("W", 20))
    H = int(cfg["H"]) if cfg.get("H") is not None else int(train_cfg.get("H", 1))
    bs = int(cfg["batch_size"])
    first_k = cfg.get("first_k")

    with open(test_list) as f:
        files = [Path(line.strip()) for line in f if line.strip()]

    for p in files:
        out_json = out_dir / f"{p.stem}_pred_gru.json"
        infer_file(str(p), out_json, model, cap_vec, device, W, H, bs, first_k=first_k)

if __name__ == "__main__":
    main()