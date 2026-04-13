from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


def plot_one_route(
    npz_path: Path,
    save_path: Path,
    epsilon: float,
    use_z_after: bool = True,
) -> None:
    data = np.load(npz_path, allow_pickle=True)

    required_keys = ["chosen_rate", "global_best_rate"]
    for key in required_keys:
        if key not in data:
            raise KeyError(f"{npz_path} missing required key: {key}")

    chosen = np.asarray(data["chosen_rate"]).astype(float)
    best = np.asarray(data["global_best_rate"]).astype(float)

    if chosen.ndim != 1 or best.ndim != 1:
        raise ValueError(f"{npz_path} chosen_rate/global_best_rate must be 1D")

    if len(chosen) != len(best):
        raise ValueError(f"{npz_path} chosen_rate and global_best_rate length mismatch")

    if "frame_idx" in data:
        x = np.asarray(data["frame_idx"]).astype(int)
    else:
        x = np.arange(len(chosen))

    if use_z_after and "z_after" in data:
        z = np.asarray(data["z_after"]).astype(float)
    elif "z_before" in data:
        z = np.asarray(data["z_before"]).astype(float)
    elif "z_after" in data:
        z = np.asarray(data["z_after"]).astype(float)
    else:
        raise KeyError(
            f"{npz_path} does not contain z_after or z_before. "
            "This script is for the new risk inference output."
        )

    ratio = np.divide(
        chosen,
        best,
        out=np.zeros_like(chosen, dtype=float),
        where=best > 0,
    )

    hit_best = np.isclose(chosen, best, rtol=1e-6, atol=1e-8)

    route_name = npz_path.stem

    # 对齐长度，防止极少数情况下字段长度略有差异
    m = min(len(x), len(chosen), len(best), len(z))
    x = x[:m]
    chosen = chosen[:m]
    best = best[:m]
    z = z[:m]
    ratio = ratio[:m]
    hit_best = hit_best[:m]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # ---------- 上图：chosen vs ground truth ----------
    ax0 = axes[0]
    ax0.plot(x, chosen, label="chosen capacity")
    ax0.plot(x, best, label="ground truth")
    ax0.set_ylabel("Capacity")
    ax0.set_title(route_name)
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    summary_text = (
        f"T={m} | "
        f"mean chosen={np.mean(chosen):.4f} | "
        f"mean gt={np.mean(best):.4f} | "
        f"mean ratio={np.mean(ratio):.4f} | "
        f"hit best={np.mean(hit_best):.4f}"
    )
    ax0.text(
        0.01, 0.98, summary_text,
        transform=ax0.transAxes,
        ha="left", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        fontsize=9,
    )

    # ---------- 下图：z ----------
    ax1 = axes[1]
    ax1.plot(x, z, label="z")
    threshold = 1.0 + float(epsilon)
    ax1.axhline(threshold, linestyle="--", label=f"threshold={threshold:.2f}")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("z")
    ax1.set_title("Risk state")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_all_routes(
    input_dir: Path,
    output_dir: Path,
    epsilon: float,
    glob_pattern: str = "*.npz",
) -> None:
    files = sorted(input_dir.glob(glob_pattern))
    if not files:
        raise RuntimeError(f"No npz files found in {input_dir} with pattern {glob_pattern}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for npz_path in tqdm(files, desc="Plotting routes"):
        out_png = output_dir / f"{npz_path.stem}.png"
        plot_one_route(
            npz_path=npz_path,
            save_path=out_png,
            epsilon=epsilon,
            use_z_after=True,
        )


def main() -> None:
    # ============================================================
    # Edit config here
    # ============================================================
    
    epsilon = 0.233572
    run_dir = "run_Ratio_alpha_0.1_eps_0.233572_horizon_256_20260410_143901"
    
    input_dir = Path(f"./checkpoints_recurrent_ppo_risk/temp/{run_dir}/inference_fullroute_best_test/routes")
    output_dir = Path(f"./checkpoints_recurrent_ppo_risk/temp/{run_dir}/inference_fullroute_best_test/plots")
    glob_pattern = "*.npz"

    # ============================================================
    # Run
    # ============================================================
    plot_all_routes(
        input_dir=input_dir,
        output_dir=output_dir,
        epsilon=epsilon,
        glob_pattern=glob_pattern,
    )

    print(f"Done. Plots saved to: {output_dir}")


if __name__ == "__main__":
    main()