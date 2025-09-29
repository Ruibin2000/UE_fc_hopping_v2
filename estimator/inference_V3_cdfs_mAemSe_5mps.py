# ===================== inference_multi_npz_all_mae_mse_cdf.py (Transformer 对齐版) =====================
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from typing import Sequence, Mapping, Optional
from matplotlib.ticker import FormatStrFormatter

THIS_DIR = Path(__file__).resolve().parent
measure_time = 0.05


USER_CFG = dict(
    ckpt=str(THIS_DIR / "newBandit_multi_0.2_150_5mps_mSe_lr1e-3_W20_V2_tfmr_global.pt"),
    test_list=str(THIS_DIR / "test_0.2_150_5mps.txt"),
    batch_size=512,
    device=f"cuda:{2}" if torch.cuda.is_available() else "cpu",
    out_dir=str(THIS_DIR / "figs_loss_mSe_0.2_150_5mps_V3_global"),
    W=None,
    H=None,
    first_k=None,
)

def parse_args() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str)
    p.add_argument("--test_list", type=str)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--device", type=str)
    p.add_argument("--out_dir", type=str)
    p.add_argument("--W", type=int)
    p.add_argument("--H", type=int)
    p.add_argument("--first_k", type=int)
    return {k: v for k, v in vars(p.parse_args()).items() if v is not None}

try:
    from dtw_tools import run_dtw_analysis, CONFIG as DTW_CONFIG  # noqa: F401
except Exception:
    run_dtw_analysis, DTW_CONFIG = None, None  # 静默忽略

def reduce_to_1d(y_mat: np.ndarray, mode: str = "max", antenna: int | None = None) -> np.ndarray:
    if y_mat.ndim != 2:
        raise ValueError(f"y_mat must be (S,N), got {y_mat.shape}")
    if mode == "max":   return y_mat.max(axis=1)
    if mode == "mean":  return y_mat.mean(axis=1)
    if mode == "ant":
        assert antenna is not None and 0 <= antenna < y_mat.shape[1], "invalid antenna index"
        return y_mat[:, antenna]
    raise ValueError(f"Unknown reduce mode: {mode}")

def build_history_features(RM_hist: torch.Tensor, SM_hist: torch.Tensor, M_hist: torch.Tensor) -> torch.Tensor:
    return torch.stack([RM_hist, SM_hist, M_hist], dim=-1)  # (W,N,3)

class MasksOnlyDatasetNext(Dataset):
    def __init__(self, npz_path: str, W: int, H: int, cap_vec: Optional[np.ndarray]):
        data = np.load(npz_path)
        rates = data["rates"].astype(np.float32)
        rm    = data["rate_mask"].astype(np.float32)
        self.sm   = data["snr_mask"].astype(np.float32)
        self.mask = data["mask"].astype(np.float32)

        if cap_vec is not None:
            cap_vec = cap_vec.astype(np.float32)
            rates = np.clip(rates, 0.0, cap_vec[None, :])
            rm    = np.clip(rm,    0.0, cap_vec[None, :])

        self.rates = rates
        self.rm    = rm
        self.T, self.N = self.rates.shape
        self.W, self.H = int(W), int(H)
        if self.T <= self.W + self.H:
            raise ValueError("T must > W+H")

    def __len__(self): 
        return self.T - self.W - self.H + 1

    def __getitem__(self, idx: int):
        s, e = idx, idx + self.W
        t = e - 1; tp1 = t + self.H
        RM_hist = torch.tensor(self.rm[s:e],   dtype=torch.float32)
        SM_hist = torch.tensor(self.sm[s:e],   dtype=torch.float32)
        M_hist  = torch.tensor(self.mask[s:e], dtype=torch.float32)
        R_tp1   = torch.tensor(self.rates[tp1], dtype=torch.float32)  # 物理值
        X = build_history_features(RM_hist, SM_hist, M_hist)
        return X, R_tp1, tp1

class LearnableAffine(nn.Module):
    def __init__(self, feat_dim: int, min_alpha: float = 1e-3):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(feat_dim))
        self.beta  = nn.Parameter(torch.zeros(feat_dim))
        self.min_alpha = float(min_alpha)
    def forward(self, x):  # (..., F)
        alpha = self.alpha.clamp_min(self.min_alpha)
        return x * alpha + self.beta

class TemporalConvEncoder(nn.Module):
    """
    Input:  (B, N, W, C)   [C = 2F: raw + scaled]
    Output: (B, N, D)
    """
    def __init__(self, in_channels: int, hidden: int = 64, out_dim: int = 64, kernel_size: int = 5):
        super().__init__()
        pad = kernel_size // 2
        self.dw = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size, padding=pad, groups=in_channels)
        self.pw1 = nn.Conv1d(in_channels, hidden, kernel_size=1)
        self.act = nn.GELU()
        self.ln  = nn.LayerNorm(hidden)  # on (B*N, W, hidden)
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
            nn.Softplus(beta=1.5)
        )

    def _apply_cap(self, y: torch.Tensor) -> torch.Tensor:
        if not self.use_cap or self.cap_vec is None:
            return y
        if self.use_softcap:
            denom = self.cap_vec[None, :] * self.softcap_sharp + 1e-6
            return self.cap_vec[None, :] * torch.sigmoid(y / denom)
        else:
            return torch.minimum(y, self.cap_vec[None, :])

    def forward(self, X_hist):  # X_hist: (B, W, N, F)
        X_hist = torch.nan_to_num(X_hist)
        B, W, N, F = X_hist.shape
        assert N == self.N and F == self.F, f"Expect N={self.N},F={self.F}, got N={N},F={F}"

        X_scaled = self.scaler(X_hist)                   # (B,W,N,F)
        x = torch.cat([X_hist, X_scaled], dim=-1)        # (B,W,N,2F)

        x_perm = x.permute(0, 2, 1, 3).contiguous()      # (B,N,W,2F)
        z = self.temporal(x_perm)                        # (B,N,z_dim)

        z = self.interact(z, groups=self.groups)         # (B,N,z_dim)

        raw_mean = X_hist.mean(dim=1)                    # (B,N,F)
        h = torch.cat([z, raw_mean], dim=-1)             # (B,N,z_dim+F)
        y = self.head(h).squeeze(-1)                     # (B,N)
        y = self._apply_cap(y)
        return {"R_hat": y}

# ----------------- load ckpt -----------------
def load_ckpt(ckpt_path: str, device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    N       = int(ckpt["N"])
    F       = int(ckpt.get("feat_dim", 3))
    z_dim   = int(ckpt.get("z_dim", 128))
    cfg     = ckpt.get("cfg", {})  
    dropout       = float(cfg.get("dropout", 0.2))
    use_softcap   = bool(cfg.get("use_softcap", False))
    softcap_sharp = float(cfg.get("softcap_sharp", 1.0))
    num_heads     = int(cfg.get("num_heads", 4))
    attn_mode     = str(cfg.get("attn_mode", "grouped"))
    attn_bias     = float(cfg.get("attn_bias", 0.5))
    use_ant_id    = bool(cfg.get("use_ant_id", True))
    ffn_mult      = int(cfg.get("ffn_mult", 4))
    cap_vec       = np.asarray(ckpt["cap_vec"], dtype=np.float32)

    model = RateEstimatorV2(
        num_antennas=N, feat_dim=F, z_dim=z_dim, dropout=dropout,
        use_cap=True, use_softcap=use_softcap, softcap_sharp=softcap_sharp,
        groups=None, num_heads=num_heads,
        attn_mode=attn_mode, attn_bias=attn_bias, use_ant_id=use_ant_id, ffn_mult=ffn_mult
    ).to(device)

    state_dict = ckpt["model"]
    model_keys = set(model.state_dict().keys())
    trimmed = {k: v for k, v in state_dict.items() if k in model_keys}

    unexpected_keys = [k for k in state_dict.keys() if k not in model_keys]
    missing_keys = [k for k in model_keys if k not in trimmed]
    if unexpected_keys:
        print("[load_ckpt] drop unexpected keys:", unexpected_keys)
    if missing_keys:
        print("[load_ckpt] model missing keys (keep init):", missing_keys)

    model.load_state_dict(trimmed, strict=False)

    with torch.no_grad():
        model.cap_vec.copy_(torch.tensor(cap_vec, device=device))
    model.eval()
    return model, cap_vec, device, cfg

# ----------------- baseline: masked previous -----------------
def generate_masked_R_prev(R_true: np.ndarray, mask: np.ndarray, W: int) -> np.ndarray:
    masked_R_prev = [R_true[0].tolist()]
    for idx in range(1, len(R_true)):
        masked_R_prev.append((1 - mask[idx - 1]) * masked_R_prev[-1] + mask[idx - 1] * R_true[idx - 1])
    return np.array(masked_R_prev[W:])

# ----------------- general CDF figure-----------------
def plot_cdf_three_series(
    series: Dict[str, np.ndarray],
    save_path: Path,
    title: str,
    xlabel: str = "MSE",
    colors: Optional[List[str]] = None, 
    alphas: Optional[List[str]] = None, 
):
    fig, ax = plt.subplots(figsize=(7, 3))
    idx = 0
    for label, arr in series.items():
        arr = np.asarray(arr, dtype=np.float64)
        arr_sorted = np.sort(arr)
        y = np.linspace(0.0, 1.0, len(arr_sorted), endpoint=False)
        ax.plot(arr_sorted, y, label=label, linewidth=2, c = colors[idx], alpha = alphas[idx])
        idx = idx + 1
    ax.set_title(title, fontsize=18)
    # ax.set_xlabel(xlabel, fontsize=16)
    # ax.xaxis.set_major_formatter(FormatStrFormatter("%.2e"))
    ax.set_ylabel("CDF", fontsize=16)
    # ax.set_xscale("log")
    ax.set_xlim(-10000, 200000)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    # ax.legend(loc="lower right", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

# ----------------- per-time MSE -----------------
@torch.no_grad()
def compute_errors_per_file(
    npz_path: str, W: int, H: int,
    model: nn.Module, cap_vec: np.ndarray, device: str,
    batch_size: int = 512
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    return:
      mae_model_t, mae_full_prev_t, mae_masked_prev_t,
      mse_model_t, mse_full_prev_t, mse_masked_prev_t   # 均为 shape (S,)
    """
    raw = np.load(npz_path)
    rates = raw["rates"].astype(np.float32)      # (T, N)
    rm    = raw["rate_mask"].astype(np.float32)
    sm    = raw["snr_mask"].astype(np.float32)
    m     = raw["mask"].astype(np.float32)
    T, N = rates.shape

    class RawDataset(Dataset):
        def __len__(self): return T - W - H + 1
        def __getitem__(self, idx):
            s, e = idx, idx + W
            t = e - 1
            tp = t + H
            RM_hist = torch.tensor(rm[s:e], dtype=torch.float32)
            SM_hist = torch.tensor(sm[s:e], dtype=torch.float32)
            M_hist  = torch.tensor(m[s:e],  dtype=torch.float32)
            X_hist  = torch.stack([RM_hist, SM_hist, M_hist], dim=-1)  # (W,N,3)
            R_tp    = torch.tensor(rates[tp], dtype=torch.float32)
            return X_hist, R_tp, tp

    loader = DataLoader(RawDataset(), batch_size=batch_size, shuffle=False)

    ts_all, pred_all = [], []
    for X_hist, _R_tp, tp in loader:
        X_hist = X_hist.to(device)
        out = model(X_hist)
        Rhat_phys = out["R_hat"].detach().cpu().numpy()  # (B, N)
        pred_all.append(Rhat_phys)
        ts_all.append(tp.numpy())

    ts = np.concatenate(ts_all, 0)
    order = np.argsort(ts)
    ts = ts[order]
    y_pred = np.concatenate(pred_all, 0)[order]   # (S, N)
    y_true = rates[ts]                             # (S, N)

    # full previous baseline:  R[t-1]
    ts_prev = ts - 1
    valid = ts_prev >= 0
    y_true_u = y_true[valid]
    y_full_prev = rates[ts_prev[valid]]

    # masked previous baseline
    y_masked_prev_all = generate_masked_R_prev(rates, m, W)  # (S, N)

    # MAE/MSE over antennas (mean)
    mae_model_t = np.abs(y_pred - y_true).mean(axis=1)                 # (S,)
    mae_full_prev_t = np.abs(y_full_prev - y_true_u).mean(axis=1)      # (S_valid,)
    mae_masked_prev_t = np.abs(y_masked_prev_all - y_true).mean(axis=1)

    mse_model_t = ((y_pred - y_true) ** 2).mean(axis=1)                # (S,)
    mse_full_prev_t = ((y_full_prev - y_true_u) ** 2).mean(axis=1)     # (S_valid,)
    mse_masked_prev_t = ((y_masked_prev_all - y_true) ** 2).mean(axis=1)

    mae_model_t = mae_model_t[valid]
    mse_model_t = mse_model_t[valid]
    mae_masked_prev_t = mae_masked_prev_t[valid]
    mse_masked_prev_t = mse_masked_prev_t[valid]

    return (mae_model_t, mae_full_prev_t, mae_masked_prev_t,
            mse_model_t, mse_full_prev_t, mse_masked_prev_t)

@torch.no_grad()
def plot_per_antenna_mSe_for_file(
    npz_path: str, W: int, H: int,
    model: nn.Module, cap_vec: np.ndarray, device: str,
    out_dir: Path, first_k: Optional[int] = None
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) load data
    raw = np.load(npz_path)
    rates = raw["rates"].astype(np.float32)      # (T, N)
    m     = raw["mask"].astype(np.float32)       # (T, N)
    T, N  = rates.shape

    # 2) dataset
    ds = MasksOnlyDatasetNext(npz_path, W=W, H=H, cap_vec=cap_vec)
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    ts_all, pred_all = [], []
    for X, _R_true, tp1 in loader:
        X = X.to(device)
        Rhat = model(X)["R_hat"].detach().cpu().numpy()  # (B, N)
        pred_all.append(Rhat)
        ts_all.append(tp1.numpy())

    ts = np.concatenate(ts_all, axis=0)
    order = np.argsort(ts)
    ts = ts[order]                                   # (S,)
    y_pred = np.concatenate(pred_all, axis=0)[order] # (S, N)
    y_true = rates[ts]                               # (S, N)
    y_mask = m[ts]                                   # (S, N)

    # 3) masked previous
    y_masked_prev_full = generate_masked_R_prev(rates, m, W)  # (T-W, N)
    idx = ts - W
    assert np.all((idx >= 0) & (idx < y_masked_prev_full.shape[0]))
    y_masked_prev = y_masked_prev_full[idx]                   # (S, N)

    # 4) first_k, optional
    if first_k is not None:
        K = int(min(first_k, len(ts)))
        ts, y_pred, y_true, y_mask, y_masked_prev = (
            ts[:K], y_pred[:K], y_true[:K], y_mask[:K], y_masked_prev[:K]
        )

    # 5) draw
    fig, axes = plt.subplots(nrows=N + 1, ncols=1, figsize=(12, 3 * (N + 1)), sharex=True)
    if N + 1 == 1:
        axes = [axes]

    t_min = ts * measure_time  # seconds

    ax0 = axes[0]
    y_true_max        = y_true.max(axis=1)
    y_pred_max        = y_pred.max(axis=1)
    y_masked_prev_max = y_masked_prev.max(axis=1)

    ax0.plot(t_min, y_true_max,              label="max true",        linewidth=1.2, c = "#2ca02c", alpha = 1)
    ax0.plot(t_min, y_masked_prev_max,       label="max masked prev",   linewidth=1.5, c = "#1f77b4", alpha = 0.9)
    ax0.plot(t_min, y_pred_max,              label="max proposed model", linewidth=1.2, c = "#ff7f0e", alpha = 1)

    mse_max_model = float(((y_true_max - y_pred_max) ** 2).mean())
    mse_max_fullPrev = float(((y_true_max - y_masked_prev_max) ** 2).mean())
    ax0.set_ylabel("Capacity (Mbps)",fontsize=18) 
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="lower right",fontsize=16)
    ax0.set_title(f"Max Predicted Capacity VS Max True Capacity over all antennas",fontsize=18)
    ax0.tick_params(axis="both", labelsize=14)
    ax0.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))

    se_model = (y_pred - y_true) ** 2            # (S, N)
    se_maskp = (y_masked_prev - y_true) ** 2     # (S, N)

    for ant in range(N):
        ax = axes[ant + 1]
        idx_obs = np.flatnonzero(y_mask[:, ant] > 0.5)
        if idx_obs.size > 0:
            ymin, ymax = 0, max(np.max(se_maskp[:, ant]), np.max(se_model[:, ant]))
            ax.vlines(t_min[idx_obs], ymin=ymin, ymax=ymax, colors="0.7", alpha=0.3, linewidth=1)
            ax.plot([], [], color="0.5", alpha=0.35)
            
        ax.plot(t_min, se_maskp[:, ant], label=f"masked prev", linewidth=1.5, c = "#1f77b4", alpha = 0.9)
        ax.plot(t_min, se_model[:, ant], label=f"proposed model", c = "#ff7f0e", linewidth=1.5, alpha = 1)

        mse_model_mean = float(se_model[:, ant].mean())
        mse_maskp_mean = float(se_maskp[:, ant].mean())
        ax.set_ylabel("MSE",fontsize=18)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2e"))
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", labelsize=14)
        if ant == 0:
            ax.legend(loc="upper right",fontsize=16)
        ax.set_title(f"Antenna {ant} | average MSE: proposed model={mse_model_mean:.3f}, masked prev={mse_maskp_mean:.3f}",fontsize=18)

    axes[-1].set_xlabel("time (seconds)",fontsize=18)
    # fig.suptitle(Path(npz_path).name, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = out_dir / f"{Path(npz_path).stem}_per_antenna_mSe.png"
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def main():
    cfg = USER_CFG.copy()
    cfg.update(parse_args())

    CKPT = Path(cfg["ckpt"])
    TEST_LIST = Path(cfg["test_list"])
    assert CKPT.exists(), f"ckpt not found: {CKPT}"
    assert TEST_LIST.exists(), f"test list not found: {TEST_LIST}"

    model, cap_vec, device, train_cfg = load_ckpt(str(CKPT), cfg.get("device"))
    W = int(cfg["W"]) if cfg.get("W") is not None else int(train_cfg.get("W", 20))
    H = int(cfg["H"]) if cfg.get("H") is not None else int(train_cfg.get("H", 1))
    batch_size = int(cfg["batch_size"])
    out_dir = Path(cfg["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)

    with open(TEST_LIST) as f:
        test_files = [Path(line.strip()) for line in f if line.strip()]

    agg_mae_model, agg_mae_full, agg_mae_masked = [], [], []
    agg_mse_model, agg_mse_full, agg_mse_masked = [], [], []

    for p in test_files:
        (mae_m, mae_f, mae_ms,
         mse_m, mse_f, mse_ms) = compute_errors_per_file(
            npz_path=str(p), W=W, H=H,
            model=model, cap_vec=cap_vec, device=device,
            batch_size=batch_size
        )

        save_mse = out_dir / f"{p.stem}_cdf_mse.png"
        plot_cdf_three_series(
            {"Full prev": mse_f, "Masked prev": mse_ms, "Proposed model": mse_m},
            colors=["#17becf", "#1f77b4", "#ff7f0e"],
            alphas=[0.8, 1, 1],
            save_path=save_mse, 
            title=f"{p.name} — CDF of per-time MSE"
        )
        
        plot_per_antenna_mSe_for_file(
            npz_path=str(p), W=W, H=H,
            model=model, cap_vec=cap_vec, device=device,
            out_dir=out_dir, first_k=cfg.get("first_k")
        )
        agg_mae_model.append(mae_m);   agg_mae_full.append(mae_f);   agg_mae_masked.append(mae_ms)
        agg_mse_model.append(mse_m);   agg_mse_full.append(mse_f);   agg_mse_masked.append(mse_ms)

        print(f"[{p.name}] mean MAE: model={mae_m.mean():.6f}, full_prev={mae_f.mean():.6f}, masked_prev={mae_ms.mean():.6f}")
        print(f"[{p.name}] mean MSE: model={mse_m.mean():.6f}, full_prev={mse_f.mean():.6f}, masked_prev={mse_ms.mean():.6f}")

    if agg_mae_model:
        all_mae_model  = np.concatenate(agg_mae_model,  axis=0)
        all_mae_full   = np.concatenate(agg_mae_full,   axis=0)
        all_mae_masked = np.concatenate(agg_mae_masked, axis=0)
        all_mse_model  = np.concatenate(agg_mse_model,  axis=0)
        all_mse_full   = np.concatenate(agg_mse_full,   axis=0)
        all_mse_masked = np.concatenate(agg_mse_masked, axis=0)

        save_all_mse = out_dir / "ALL_files_cdf_mse.png"
        plot_cdf_three_series(
            {
                "Full prev": all_mse_full,
                "Masked prev": all_mse_masked,
                "Proposed model": all_mse_model,
            },
            colors=["#1f77b4", "#2ca02c", "#ff7f0e"],
            alphas=[1, 1.0, 1.0],
            save_path=save_all_mse,
            title="Medium Mobility"
        )

        print(f"[ALL] mean MAE: model={all_mae_model.mean():.6f}, full_prev={all_mae_full.mean():.6f}, masked_prev={all_mae_masked.mean():.6f}")
        print(f"[ALL] mean MSE: model={all_mse_model.mean():.6f}, full_prev={all_mse_full.mean():.6f}, masked_prev={all_mse_masked.mean():.6f}")

if __name__ == "__main__":
    main()
