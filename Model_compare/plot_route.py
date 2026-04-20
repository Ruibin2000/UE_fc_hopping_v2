from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

ROUTE_ID = 947

RAW_DIR = Path("../dataset/CQI_arr8_medium_R_2_region_3")
PPO_DIR = Path("../PPO/checkpoints_v5/PPO_G4_base_risk_log1p_alpha0.1_eps0.3_lam10.3_lam2600.0_alr9e-05_clr5e-05_20260416_174318/inference_eval/test_best/route_npz")
BANDIT_DIR = Path("../benchmark/benchmark_bandit/v5_results/G4_L1_base_test_risk_aware_semi_bandit_alphaQ_0.1_eps_0.3_riskA_0.1_budget_0.3_lam1_0.3_lam2_600.0_benchLam1_None_benchLam2_None_rewardMode_risk_log1p_hardConst_0_20260417_183454/routes")
SAVE_DIR = Path("./per_route_plots")

# x-axis display range; None = full route
# X_RANGE = None
X_RANGE = (500, 800)

# fixed y-axis ranges; None = let matplotlib decide
Y_RAW = None
Y_Z = None
Y_RATE = None
# example:
# Y_RAW = (0, 700)
# Y_Z = (1.0, 2.0)
# Y_RATE = (0, 700)

import re

import re

def parse_explore_rate_from_path(path: Path):
    s = str(path)
    m = re.search(r"eps([0-9.]+)", s)
    if m:
        return float(m.group(1))
    return None

EXPLORE_RATE = parse_explore_rate_from_path(PPO_DIR)
# ============================================================
# Auto paths
# ============================================================

route_id_str = f"{ROUTE_ID:04d}"

RAW_ROUTE_FILE = RAW_DIR / f"routes_{route_id_str}_result.npz"
PPO_ROUTE_FILE = PPO_DIR / f"ppo_routes_{route_id_str}_result.npz"
BANDIT_ROUTE_FILE = BANDIT_DIR / f"bandit_routes_{route_id_str}_result.npz"

SAVE_PATH = SAVE_DIR / f"compare_routes_{route_id_str}.png"


# ============================================================
# Load
# ============================================================

def load_route_capacity(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)

    if "capacity" not in data:
        raise KeyError(f"'capacity' not found in {npz_path}")

    cap = np.squeeze(np.array(data["capacity"])).astype(float)

    if cap.ndim != 3:
        raise ValueError(f"Expected 3D capacity, got {cap.shape} in {npz_path}")

    cap = cap.transpose(0, 2, 1)   # [T, N_cell, N_arr]
    return cap.astype(np.float32)


def load_method(npz_path: Path) -> dict:
    data = np.load(npz_path, allow_pickle=True)

    if "chosen_rate" not in data:
        raise KeyError(f"{npz_path} missing chosen_rate")

    if "z_trace" in data:
        z = data["z_trace"]          # PPO
    elif "z_before" in data:
        z = data["z_before"]         # Bandit
    elif "z_after" in data:
        z = data["z_after"]
    else:
        raise KeyError(f"{npz_path} missing z key. keys={list(data.files)}")

    return {
        "chosen": np.asarray(data["chosen_rate"], dtype=float),
        "z": np.asarray(z, dtype=float),
        "best": np.asarray(data["global_best_rate"], dtype=float)
        if "global_best_rate" in data else None,
    }


# ============================================================
# Plot
# ============================================================

def plot_compare(
    route: np.ndarray,
    ppo: dict,
    bandit: dict,
    save_path: Path,
    x_range=None,
    y_raw=None,
    y_z=None,
    y_rate=None,
) -> None:
    T, n_cell, n_arr = route.shape
    x = np.arange(T)

    if len(ppo["chosen"]) != T or len(ppo["z"]) != T:
        raise ValueError("PPO length does not match raw route length")
    if len(bandit["chosen"]) != T or len(bandit["z"]) != T:
        raise ValueError("Bandit length does not match raw route length")
    if ppo["best"] is not None and len(ppo["best"]) != T:
        raise ValueError("PPO best length does not match raw route length")

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 2, 2]},
    )

    # 1) top: raw
    colors = plt.get_cmap("tab10")(np.arange(3, 10))  # 跳过蓝橙

    ax1.set_prop_cycle(color=colors)

    for cell in range(n_cell):
        for arr in range(n_arr):
            ax1.plot(x, route[:, cell, arr], linewidth=1.5)
    
    ax1.set_ylabel("Raw")
    ax1.set_title(f"Route {route_id_str}")
    ax1.grid(True, alpha=0.3)

    # 2) middle: z compare
    ax2.plot(x, bandit["z"], "--", color="tab:blue", linewidth=2.0)
    ax2.plot(x, ppo["z"], "-", color="tab:orange", linewidth=2.0)
    ax2.axhline(
        y=1+EXPLORE_RATE,
        color="black",
        linestyle=":",
        linewidth=2.0,
        label=r"Threshold (1 +  $\epsilon$)",
    )

    ax2.set_ylabel("z")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")

    # 3) bottom: chosen compare
    ax3.plot(x, ppo["chosen"], "-", color="tab:orange",  label="PPO risk-aware", linewidth=2.0)
    ax3.plot(x, bandit["chosen"], "--", color="tab:blue", label="Bandit", linewidth=1.5)

    
    if ppo["best"] is not None:
        ax3.plot(x, ppo["best"], "-", color="tab:green", label="Best", linewidth=1.5)

    ax3.set_ylabel("Chosen Rate")
    ax3.set_xlabel("Time step")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="lower right")

    # fixed x range
    if x_range is not None:
        ax1.set_xlim(x_range)

    # fixed y ranges
    if y_raw is not None:
        ax1.set_ylim(y_raw)
    if y_z is not None:
        ax2.set_ylim(y_z)
    if y_rate is not None:
        ax3.set_ylim(y_rate)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main() -> None:
    route = load_route_capacity(RAW_ROUTE_FILE)
    ppo = load_method(PPO_ROUTE_FILE)
    bandit = load_method(BANDIT_ROUTE_FILE)


    if EXPLORE_RATE is None:
        raise ValueError("Cannot find explore rate from PPO_DIR")

    print("Explore rate:", EXPLORE_RATE)

    plot_compare(
        
        route=route,
        ppo=ppo,
        bandit=bandit,
        save_path=SAVE_PATH,
        x_range=X_RANGE,
        y_raw=Y_RAW,
        y_z=Y_Z,
        y_rate=Y_RATE,
    )

    print(f"Saved: {SAVE_PATH}")


if __name__ == "__main__":
    main()