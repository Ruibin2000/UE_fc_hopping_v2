# ===================== inference_lstm_save_json.py =====================
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
    ckpt=str(THIS_DIR / "newBandit_multi_0.2_150_") +mobility+ str("_LSTM_mSe_lr1e-3_W20_batch_512.pt"),
    test_list=str(THIS_DIR / "test_0.2_150_") +mobility+ ".txt",
    out_dir=str(THIS_DIR / "pred_json_LSTM"),
    batch_size=512,
    device=f"cuda:{0}" if torch.cuda.is_available() else "cpu",
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
        self.T, self.N = self.rates.shape
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

class LearnableAffine(nn.Module):
    def __init__(self, feat_dim: int, min_alpha: float = 1e-3):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(feat_dim))
        self.beta  = nn.Parameter(torch.zeros(feat_dim))
        self.min_alpha = float(min_alpha)
    def forward(self, x):
        alpha = self.alpha.clamp_min(self.min_alpha)
        return x * alpha + self.beta

class RateEstimatorLSTM(nn.Module):
    def __init__(self, num_antennas: int, feat_dim: int = 3,
                 hidden: int = 128, num_layers: int = 2, dropout: float = 0.2,
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
        self.scaler = LearnableAffine(self.F)

        self.lstm = nn.LSTM(
            input_size=self.F,
            hidden_size=self.hidden,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(self.hidden + self.F),
            nn.Linear(self.hidden + self.F, self.hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden, 1),
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

    def forward(self, X_hist: torch.Tensor):
        X_hist = torch.nan_to_num(X_hist)
        B, W, N, F = X_hist.shape
        X_scaled = self.scaler(X_hist)
        x = X_scaled.permute(0, 2, 1, 3).contiguous().view(B * N, W, F)
        _, (h_n, _) = self.lstm(x)
        h_last = h_n[-1]  # (B*N, hidden)
        raw_mean = torch.nan_to_num(X_hist.mean(dim=1)).contiguous().view(B * N, F)
        h = torch.cat([h_last, raw_mean], dim=-1)
        y = self.head(h).view(B, N)
        y = torch.nan_to_num(y)
        y = self._apply_cap(y)
        return {"R_hat": y}

def load_ckpt_lstm(ckpt_path: str, device: str):
    # ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    N = int(ckpt["N"])
    F = int(ckpt.get("feat_dim", 3))
    cfg = ckpt.get("cfg", {})
    cap_vec = np.asarray(ckpt["cap_vec"], dtype=np.float32)

    hidden = int(cfg.get("hidden", ckpt.get("hidden", 128)))
    num_layers = int(cfg.get("num_layers", ckpt.get("num_layers", 2)))
    dropout = float(cfg.get("dropout", 0.2))
    use_softcap = bool(cfg.get("use_softcap", False))
    softcap_sharp = float(cfg.get("softcap_sharp", 1.0))

    model = RateEstimatorLSTM(
        num_antennas=N, feat_dim=F,
        hidden=hidden, num_layers=num_layers, dropout=dropout,
        use_cap=True, use_softcap=use_softcap, softcap_sharp=softcap_sharp
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
def infer_file_to_json(npz_path: str, out_json: Path, model: nn.Module,
                       cap_vec: np.ndarray, device: str,
                       W: int, H: int, batch_size: int,
                       first_k: Optional[int] = None):
    ds = MasksOnlyDatasetNext(npz_path, W=W, H=H, cap_vec=cap_vec)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    t_list: List[int] = []
    pred_list: List[List[float]] = []

    for X, tp1 in loader:
        X = X.to(device, non_blocking=True)
        R_hat = model(X)["R_hat"].detach().cpu().numpy()
        for i in range(R_hat.shape[0]):
            t_list.append(int(tp1[i]))
            pred_list.append(R_hat[i].astype(np.float32).tolist())
            if first_k is not None and len(t_list) >= first_k:
                break
        if first_k is not None and len(t_list) >= first_k:
            break

    payload: Dict[str, Any] = dict(
        npz=str(Path(npz_path).resolve()),
        W=int(W),
        H=int(H),
        t_index=t_list,
        R_hat=pred_list,
        cap_vec=cap_vec.astype(np.float32).tolist(),
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(payload, f)

def main():
    cfg = USER_CFG.copy()
    cfg.update(parse_args())

    ckpt = Path(cfg["ckpt"])
    test_list = Path(cfg["test_list"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    assert ckpt.exists(), f"ckpt not found: {ckpt}"
    assert test_list.exists(), f"test list not found: {test_list}"

    device = cfg["device"]
    model, cap_vec, train_cfg = load_ckpt_lstm(str(ckpt), device=device)

    W = int(cfg["W"]) if cfg.get("W") is not None else int(train_cfg.get("W", 20))
    H = int(cfg["H"]) if cfg.get("H") is not None else int(train_cfg.get("H", 1))
    bs = int(cfg["batch_size"])
    first_k = cfg.get("first_k")

    with open(test_list) as f:
        files = [Path(line.strip()) for line in f if line.strip()]

    for p in files:
        out_json = out_dir / f"{p.stem}_pred_lstm.json"
        infer_file_to_json(str(p), out_json, model, cap_vec, device, W, H, bs, first_k=first_k)
        print(f"[LSTM] saved -> {out_json}")

if __name__ == "__main__":
    main()