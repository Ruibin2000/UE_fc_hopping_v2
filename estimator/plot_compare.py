from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import argparse, json
import numpy as np
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent

USER_CFG = dict(
    test_list=str(THIS_DIR / "test_0.2_150_10mps.txt"),
    pred_dir_tfmr=str(THIS_DIR / "pred_json_tfmr"),
    pred_dir_lstm=str(THIS_DIR / "pred_json_lstm"),
    pred_dir_gru=str(THIS_DIR / "pred_json_gru"),
    out_dir=str(THIS_DIR / "figs_compare"),
    # xlim=None,        # "0,200000"
    xlim = "1000,200000", 
    # xscale="linear",          # "linear" | "log"
    xscale="log",          # "linear" | "log"
    include_baselines=0,      # 1: Full prev + Masked prev
    models="tfmr,lstm,gru",   # comma list subset
    strict=0,
)

def parse_args() -> dict:
    p = argparse.ArgumentParser()
    p.add_argument("--test_list", type=str)
    p.add_argument("--pred_dir_tfmr", type=str)
    p.add_argument("--pred_dir_lstm", type=str)
    p.add_argument("--pred_dir_gru", type=str)
    p.add_argument("--out_dir", type=str)
    p.add_argument("--xlim", type=str, help="e.g. 0,200000")
    p.add_argument("--xscale", type=str, choices=["linear", "log"])
    p.add_argument("--include_baselines", type=int, choices=[0, 1])
    p.add_argument("--models", type=str, help="comma list: tfmr,lstm,gru (any subset)")
    p.add_argument("--strict", type=int, choices=[0, 1])
    args = vars(p.parse_args())
    args = {k: v for k, v in args.items() if v is not None}
    return args

def load_pred_json(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        d = json.load(f)
    d["t_index"] = np.asarray(d["t_index"], dtype=np.int64)
    d["R_hat"]   = np.asarray(d["R_hat"], dtype=np.float32)  # (S,N)
    d["cap_vec"] = np.asarray(d.get("cap_vec", []), dtype=np.float32)
    d["W"] = int(d.get("W", 20))
    d["H"] = int(d.get("H", 1))
    return d

def _maybe_load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return load_pred_json(path)

def masked_prev_series(rates: np.ndarray, mask: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """
    prev[0] = rates[0]
    prev[t] = (1-mask[t-1])*prev[t-1] + mask[t-1]*rates[t-1]
    return prev[ts]
    """
    T, N = rates.shape
    prev = np.zeros((T, N), dtype=np.float32)
    prev[0] = rates[0]
    for t in range(1, T):
        prev[t] = (1.0 - mask[t - 1]) * prev[t - 1] + mask[t - 1] * rates[t - 1]
    return prev[ts]

def full_prev_series(rates: np.ndarray, ts: np.ndarray) -> np.ndarray:
    return rates[ts - 1]

def cdf_xy(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    a = np.asarray(arr, dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.array([0.0]), np.array([0.0])
    a = np.sort(a)
    y = np.linspace(0.0, 1.0, len(a), endpoint=False)
    return a, y

def plot_cdf(
    series: Dict[str, np.ndarray],
    save_path: Path,
    title: str,
    xlim: Optional[Tuple[float, float]] = None,
    xscale: str = "linear",
):
    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    # log safety: replace <=0 with eps
    eps = 1e-12
    series_plot = {}
    if xscale == "log":
        for k, v in series.items():
            vv = np.asarray(v, dtype=np.float64)
            vv = np.where(vv <= 0, eps, vv)
            series_plot[k] = vv
    else:
        series_plot = series

    for label, arr in series_plot.items():
        x, y = cdf_xy(arr)
        ax.plot(x, y, label=label, linewidth=2)

    ax.set_title(title, fontsize=14)
    ax.set_ylabel("CDF", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", labelsize=11)
    ax.set_xscale(xscale)

    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right", fontsize=10)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

def main():
    cfg = USER_CFG.copy()
    cfg.update(parse_args())

    include_baselines = bool(int(cfg["include_baselines"]))
    strict = bool(int(cfg["strict"]))
    xscale = str(cfg["xscale"])

    models_req = [s.strip().lower() for s in str(cfg["models"]).split(",") if s.strip() != ""]
    valid_models = {"tfmr", "lstm", "gru"}
    for m in models_req:
        if m not in valid_models:
            raise ValueError(f"Unknown model in --models: {m}. Use subset of {sorted(valid_models)}")

    test_list = Path(cfg["test_list"])
    pred_dir_tfmr = Path(cfg["pred_dir_tfmr"])
    pred_dir_lstm = Path(cfg["pred_dir_lstm"])
    pred_dir_gru  = Path(cfg["pred_dir_gru"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    assert test_list.exists(), f"missing test_list: {test_list}"

    xlim = None
    if cfg.get("xlim") is not None:
        a, b = str(cfg["xlim"]).split(",")
        xlim = (float(a), float(b))

    with open(test_list) as f:
        test_files = [Path(line.strip()) for line in f if line.strip()]

    all_mse_full: List[np.ndarray] = []
    all_mse_mask: List[np.ndarray] = []
    all_mse_tfmr: List[np.ndarray] = []
    all_mse_lstm: List[np.ndarray] = []
    all_mse_gru:  List[np.ndarray] = []

    for npz_path in test_files:
        stem = npz_path.stem

        # load requested models only
        pred_tfmr = pred_lstm = pred_gru = None
        if "tfmr" in models_req:
            pred_tfmr = _maybe_load(pred_dir_tfmr / f"{stem}_pred_tfmr.json")
        if "lstm" in models_req:
            pred_lstm = _maybe_load(pred_dir_lstm / f"{stem}_pred_lstm.json")
        if "gru" in models_req:
            pred_gru  = _maybe_load(pred_dir_gru  / f"{stem}_pred_gru.json")

        if strict:
            missing = []
            if "tfmr" in models_req and pred_tfmr is None: missing.append("tfmr")
            if "lstm" in models_req and pred_lstm is None: missing.append("lstm")
            if "gru"  in models_req and pred_gru  is None: missing.append("gru")
            if missing:
                raise FileNotFoundError(f"[strict] missing {missing} json for {stem}")

        # need at least one model pred OR baselines to plot
        if pred_tfmr is None and pred_lstm is None and pred_gru is None:
            print(f"[WARN] no requested model json found for {stem}, skip.")
            continue

        # reference t_index
        ref = pred_tfmr or pred_lstm or pred_gru
        assert ref is not None
        ts = ref["t_index"]
        cap_vec = ref["cap_vec"] if ref.get("cap_vec", np.array([])).size else None

        # alignment checks
        def check_align(name: str, d: Optional[Dict[str, Any]]):
            if d is None:
                return
            if d["t_index"].shape != ts.shape or np.any(d["t_index"] != ts):
                raise ValueError(f"t_index mismatch: {name} vs ref for {stem}")
        check_align("tfmr", pred_tfmr)
        check_align("lstm", pred_lstm)
        check_align("gru",  pred_gru)

        raw = np.load(npz_path)
        rates = raw["rates"].astype(np.float32)
        mask  = raw["mask"].astype(np.float32)

        valid = ts >= 1
        ts_v = ts[valid]
        y_true = rates[ts_v]
        if cap_vec is not None and cap_vec.size == y_true.shape[1]:
            y_true = np.clip(y_true, 0.0, cap_vec[None, :])

        series: Dict[str, np.ndarray] = {}
        msg_parts = [f"[{stem}]"]

        if include_baselines:
            y_full = full_prev_series(rates, ts_v)
            y_mask = masked_prev_series(rates, mask, ts_v)
            if cap_vec is not None and cap_vec.size == y_true.shape[1]:
                y_full = np.clip(y_full, 0.0, cap_vec[None, :])
                y_mask = np.clip(y_mask, 0.0, cap_vec[None, :])
            mse_full = ((y_full - y_true) ** 2).mean(axis=1)
            mse_mask = ((y_mask - y_true) ** 2).mean(axis=1)
            series["Full prev"] = mse_full
            series["Masked prev"] = mse_mask
            all_mse_full.append(mse_full)
            all_mse_mask.append(mse_mask)
            msg_parts.append(f"full_prev={mse_full.mean():.6f}")
            msg_parts.append(f"masked_prev={mse_mask.mean():.6f}")

        if pred_tfmr is not None:
            y_tfmr = pred_tfmr["R_hat"][valid]
            mse_tfmr = ((y_tfmr - y_true) ** 2).mean(axis=1)
            series["Transformer"] = mse_tfmr
            all_mse_tfmr.append(mse_tfmr)
            msg_parts.append(f"tfmr={mse_tfmr.mean():.6f}")

        if pred_lstm is not None:
            y_lstm = pred_lstm["R_hat"][valid]
            mse_lstm = ((y_lstm - y_true) ** 2).mean(axis=1)
            series["LSTM"] = mse_lstm
            all_mse_lstm.append(mse_lstm)
            msg_parts.append(f"lstm={mse_lstm.mean():.6f}")

        if pred_gru is not None:
            y_gru = pred_gru["R_hat"][valid]
            mse_gru = ((y_gru - y_true) ** 2).mean(axis=1)
            series["GRU"] = mse_gru
            all_mse_gru.append(mse_gru)
            msg_parts.append(f"gru={mse_gru.mean():.6f}")

        if len(series) == 0:
            print(f"[WARN] nothing to plot for {stem} (maybe include_baselines=0 and no model json).")
            continue

        print(" mean MSE: ".join([msg_parts[0], ", ".join(msg_parts[1:])]))

        # plot_cdf(
        #     series=series,
        #     save_path=out_dir / f"{stem}_CDF_MSE.png",
        #     title=f"{npz_path.name} — CDF of per-time MSE",
        #     xlim=xlim,
        #     xscale=xscale,
        # )

    # ALL
    if include_baselines and len(all_mse_full) == 0 and (len(all_mse_tfmr)+len(all_mse_lstm)+len(all_mse_gru) == 0):
        print("[WARN] No plots generated.")
        return

    all_series: Dict[str, np.ndarray] = {}
    if include_baselines and len(all_mse_full) > 0:
        all_series["Full prev"]   = np.concatenate(all_mse_full, axis=0)
        all_series["Masked prev"] = np.concatenate(all_mse_mask, axis=0)

    if len(all_mse_tfmr) > 0:
        all_series["Transformer"] = np.concatenate(all_mse_tfmr, axis=0)
    if len(all_mse_lstm) > 0:
        all_series["LSTM"]        = np.concatenate(all_mse_lstm, axis=0)
    if len(all_mse_gru) > 0:
        all_series["GRU"]         = np.concatenate(all_mse_gru, axis=0)

    if len(all_series) == 0:
        print("[WARN] No ALL plot generated (no series).")
        return

    msg = "[ALL] mean MSE: "
    msg_items = []
    for k, v in all_series.items():
        msg_items.append(f"{k}={v.mean():.6f}")
    print(msg + ", ".join(msg_items))

    plot_cdf(
        series=all_series,
        save_path=out_dir / "ALL_CDF_MSE.png",
        title="ALL test files — CDF of per-time MSE",
        xlim=xlim,
        xscale=xscale,
    )

if __name__ == "__main__":
    main()