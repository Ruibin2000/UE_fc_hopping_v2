from __future__ import annotations

from pathlib import Path
import json
import logging
import re
from typing import Dict, List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Edit here
# ============================================================

group_idx = 3
benchmark_idx = 0


ppo_risk_list = [
                "PPO_G4_base_risk_log1p_alpha0.1_eps0.3_lam10.3_lam2600.0_alr9e-05_clr5e-05_20260416_174318",
                 "PPO_G4_L1_high_risk_log1p_alpha0.1_eps0.3_lam10.5_lam2600.0_alr9e-05_clr5e-05_20260417_033147",
                 "PPO_G4_L1_strong_risk_log1p_alpha0.1_eps0.3_lam10.7_lam2600.0_alr9e-05_clr5e-05_20260417_033222",
                 "PPO_G4_L1_verystrong_risk_log1p_alpha0.1_eps0.3_lam11.0_lam2600.0_alr9e-05_clr5e-05_20260417_033255",
                 ]

PPO_RISK_DIR = Path(f"../PPO/checkpoints_v5/{ppo_risk_list[group_idx]}/inference_eval/test_best/route_npz")



bandit_risk_list = [
                    "G4_L1_base_test_risk_aware_semi_bandit_alphaQ_0.1_eps_0.3_riskA_0.1_budget_0.3_lam1_0.3_lam2_600.0_benchLam1_None_benchLam2_None_rewardMode_risk_log1p_hardConst_0_20260417_183454",
                    "G4_L1_high_test_risk_aware_semi_bandit_alphaQ_0.1_eps_0.3_riskA_0.1_budget_0.3_lam1_0.5_lam2_600.0_benchLam1_None_benchLam2_None_rewardMode_risk_log1p_hardConst_0_20260417_183517",
                    "G4_L1_strong_test_risk_aware_semi_bandit_alphaQ_0.1_eps_0.3_riskA_0.1_budget_0.3_lam1_0.7_lam2_600.0_benchLam1_None_benchLam2_None_rewardMode_risk_log1p_hardConst_0_20260417_183540",
                    "G4_L1_verystrong_test_risk_aware_semi_bandit_alphaQ_0.1_eps_0.3_riskA_0.1_budget_0.3_lam1_1.0_lam2_600.0_benchLam1_None_benchLam2_None_rewardMode_risk_log1p_hardConst_0_20260417_183603"
                    ]


BANDIT_RISK_DIR = Path(
    "../benchmark/benchmark_bandit/v5_results/"
    f"{bandit_risk_list[group_idx]}"
    "/routes"
)


bandit_no_penalty_list = ["G1_lowCritic_test_risk_aware_semi_bandit_alphaQ_0.1_eps_0.3_riskA_0.1_budget_0.3_lam1_0.0_lam2_600.0_benchLam1_None_benchLam2_None_rewardMode_risk_log1p_hardConst_0_20260417_182857"]
BANDIT_NO_PENALTY_DIR = Path(
    "../benchmark/benchmark_bandit/no_penalty_bandit/"
    f"{bandit_no_penalty_list[benchmark_idx]}"
    "/routes"
)


ppo_no_penalty_list = ["PPO_G1_lowCritic_risk_log1p_alpha0.1_eps0.3_lam10.0_lam2600.0_alr9e-05_clr5e-05_20260416_174102"]
PPO_NO_PENALTY_DIR = Path(f"../PPO/checkpoints_no_penalty/{ppo_no_penalty_list[benchmark_idx]}//inference_eval/test_best/route_npz")


GROUNDTRUTH_DIR = Path("../dataset/CQI_arr8_medium_R_2_region_3")


OUTPUT_DIR = Path(f"./v5/reward_compare_G{group_idx}_benchmark_{benchmark_idx}")


# ============================================================
# Logging
# ============================================================

def build_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("reward_compare")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# ============================================================
# Helpers
# ============================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_route_id_from_name(path: Path) -> str:
    name = path.name
    m = re.search(r"routes_(\d+)_result\.npz$", name)
    if m is None:
        raise ValueError(f"Cannot extract route id from filename: {name}")
    return m.group(1)


def list_npz_files(folder: Path) -> List[Path]:
    files = sorted(folder.glob("*.npz"))
    if not files:
        raise RuntimeError(f"No .npz files found in: {folder}")
    return files


def build_route_map(files: List[Path]) -> Dict[str, Path]:
    route_map: Dict[str, Path] = {}
    for f in files:
        route_id = extract_route_id_from_name(f)
        if route_id in route_map:
            raise ValueError(f"Duplicate route id {route_id} in {f.parent}")
        route_map[route_id] = f
    return route_map


# ============================================================
# Groundtruth loading
# ============================================================

def load_groundtruth_best_rate(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)

    if "capacity" not in data:
        raise KeyError(f"'capacity' not found in {npz_path}")

    cap = np.squeeze(np.array(data["capacity"])).astype(float)
    if cap.ndim != 3:
        raise ValueError(
            f"Expected capacity to be 3D after squeeze, got {cap.shape} in {npz_path}"
        )

    cap = cap.transpose(0, 2, 1)  # [T, N_cell, N_arr]
    best_rate = cap.max(axis=(1, 2))
    return best_rate.astype(np.float64)


# ============================================================
# Inference loading
# ============================================================

def load_1d_metric_array(npz_path: Path, key: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)

    if key not in data:
        raise KeyError(f"'{key}' not found in {npz_path}")

    arr = np.asarray(data[key], dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{key} must be 1D in {npz_path}, got {arr.shape}")

    return arr


def load_chosen_rate_array(infer_path: Path) -> np.ndarray:
    return load_1d_metric_array(infer_path, "chosen_rate")


def load_z_array(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)

    # PPO
    if "z_trace" in data:
        arr = np.asarray(data["z_trace"], dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"z_trace must be 1D in {npz_path}, got {arr.shape}")
        return arr

    # bandit: 优先用 z_before
    if "z_before" in data:
        arr = np.asarray(data["z_before"], dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"z_before must be 1D in {npz_path}, got {arr.shape}")
        return arr

    if "z_after" in data:
        arr = np.asarray(data["z_after"], dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"z_after must be 1D in {npz_path}, got {arr.shape}")
        return arr

    raise KeyError(f"No z found in {npz_path}, tried: z_trace, z_before, z_after")


def load_pair_ratio_scalar(npz_path: Path) -> float:
    data = np.load(npz_path, allow_pickle=True)

    if "pair_ratio" in data:
        return float(np.asarray(data["pair_ratio"], dtype=np.float64).reshape(()))

    if "pair_action_ratio" in data:
        return float(np.asarray(data["pair_action_ratio"], dtype=np.float64).reshape(()))

    # bandit route npz fallback: 从 per-step num_open_arrays 现算
    if "num_open_arrays" in data:
        arr = np.asarray(data["num_open_arrays"], dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"num_open_arrays must be 1D in {npz_path}, got {arr.shape}")
        return float(np.mean(arr == 2))

    raise KeyError(
        f"No pair ratio found in {npz_path}, tried: pair_ratio, pair_action_ratio, num_open_arrays"
    )



# ============================================================
# Metric computation
# ============================================================

def validate_same_length(
    route_id: str,
    a: np.ndarray,
    b: np.ndarray,
    name_a: str,
    name_b: str,
) -> None:
    if len(a) != len(b):
        raise ValueError(
            f"Route {route_id}: length mismatch between {name_a} ({len(a)}) "
            f"and {name_b} ({len(b)})"
        )


def compute_hit_best_array(
    chosen_rate: np.ndarray,
    best_rate: np.ndarray,
    atol: float = 1e-8,
) -> np.ndarray:
    if chosen_rate.shape != best_rate.shape:
        raise ValueError(
            f"Shape mismatch: chosen_rate {chosen_rate.shape} vs best_rate {best_rate.shape}"
        )
    return np.isclose(chosen_rate, best_rate, atol=atol).astype(np.float64)


def compute_best_ratio_array(
    chosen_rate: np.ndarray,
    best_rate: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    if chosen_rate.shape != best_rate.shape:
        raise ValueError(
            f"Shape mismatch: chosen_rate {chosen_rate.shape} vs best_rate {best_rate.shape}"
        )

    ratio = chosen_rate / np.maximum(best_rate, eps)
    ratio = np.clip(ratio, 0.0, None)
    return ratio.astype(np.float64)

def compute_benchmark_reward_array(
    chosen_rate: np.ndarray,
    z: np.ndarray,
    lambda1: float,
    lambda2: float,
    reward_mode: str = "risk_log1p",
) -> np.ndarray:
    chosen_rate = np.asarray(chosen_rate, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)

    if chosen_rate.shape != z.shape:
        raise ValueError(
            f"Shape mismatch: chosen_rate {chosen_rate.shape} vs z {z.shape}"
        )

    if reward_mode == "rate":
        return chosen_rate.astype(np.float64)

    penalty = lambda1 * (z - 1.0) * (
        chosen_rate / (lambda2 + chosen_rate + 1e-8)
    )

    if reward_mode == "risk_aware":
        reward = chosen_rate * np.exp(-penalty)
    elif reward_mode == "risk_log1p":
        reward = np.log1p(chosen_rate) * np.exp(-penalty)
    else:
        raise ValueError(f"Unknown reward_mode: {reward_mode}")

    return reward.astype(np.float64)


# ============================================================
# Plotting
# ============================================================

def plot_CCDF_multi(
    values_dict: Dict[str, np.ndarray],
    xlabel: str,
    title: str,
    save_path: Path,
) -> None:
    plt.figure(figsize=(7, 5))
    plotted = 0

    for label, values in values_dict.items():
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue

        x = np.sort(arr)
        n = len(x)
        y = 1.0 - (np.arange(1, n + 1) / n) + (1.0 / n)   # P(X >= x)

        style_map = {
            "PPO (risk-aware)": dict(color="tab:blue", linestyle="-"),
            "PPO": dict(color="tab:blue", linestyle="--"),

            "Bandit (risk-aware)": dict(color="tab:orange", linestyle="-"),
            "Bandit": dict(color="tab:orange", linestyle="--"),

            "Optimal": dict(color="red", linestyle="-"),
        }

        style = style_map.get(label, {})

        plt.plot(
            x,
            y,
            label=label,
            linewidth=2.0,
            **style,
        )
        plotted += 1

    if plotted == 0:
        plt.close()
        raise ValueError(f"No valid values to plot for: {title}")

    plt.xlabel(xlabel)
    plt.ylabel("P(X ≥ x)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ============================================================
# Main
# ============================================================

def main() -> None:
    ensure_dir(OUTPUT_DIR)
    logger = build_logger(OUTPUT_DIR / "reward_compare.log")

    logger.info(f"PPO_RISK_DIR            : {PPO_RISK_DIR}")
    logger.info(f"PPO_NO_PENALTY_DIR      : {PPO_NO_PENALTY_DIR}")
    logger.info(f"BANDIT_RISK_DIR         : {BANDIT_RISK_DIR}")
    logger.info(f"BANDIT_NO_PENALTY_DIR   : {BANDIT_NO_PENALTY_DIR}")
    logger.info(f"GROUNDTRUTH_DIR         : {GROUNDTRUTH_DIR}")
    logger.info(f"OUTPUT_DIR              : {OUTPUT_DIR}")

    # ------------------------------------------------------------
    # Load PPO risk config
    # ------------------------------------------------------------
    config_path = PPO_RISK_DIR.parent.parent.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    lambda1 = float(cfg["lambda1"])
    lambda2 = float(cfg["lambda2"])
    epsilon_budget = float(cfg["epsilon_budget"])
    alpha_risk = float(cfg["alpha_risk"])
    reward_mode = str(cfg.get("reward_mode", "risk_log1p"))
    logger.info(
        f"Loaded config: λ1={lambda1}, λ2={lambda2}, ε={epsilon_budget}, α={alpha_risk}, reward_mode={reward_mode}"
    )

    ppo_risk_files = list_npz_files(PPO_RISK_DIR)
    ppo_no_penalty_files = list_npz_files(PPO_NO_PENALTY_DIR)
    bandit_risk_files = list_npz_files(BANDIT_RISK_DIR)
    bandit_no_penalty_files = list_npz_files(BANDIT_NO_PENALTY_DIR)
    gt_files = list_npz_files(GROUNDTRUTH_DIR)

    ppo_risk_map = build_route_map(ppo_risk_files)
    ppo_no_penalty_map = build_route_map(ppo_no_penalty_files)
    bandit_risk_map = build_route_map(bandit_risk_files)
    bandit_no_penalty_map = build_route_map(bandit_no_penalty_files)
    gt_map = build_route_map(gt_files)

    common_ids = sorted(
        set(ppo_risk_map)
        & set(ppo_no_penalty_map)
        & set(bandit_risk_map)
        & set(bandit_no_penalty_map)
        & set(gt_map)
    )
    if not common_ids:
        raise RuntimeError(
            "No common route ids found across PPO / bandit / groundtruth."
        )

    logger.info(f"Matched routes          : {len(common_ids)}")

    # reward
    ppo_risk_route_mean_reward: List[float] = []
    ppo_no_penalty_route_mean_reward: List[float] = []
    bandit_risk_route_mean_reward: List[float] = []
    bandit_no_penalty_route_mean_reward: List[float] = []
    optimal_route_mean_reward: List[float] = []

    # hit best rate
    ppo_risk_route_mean_hit_best_rate: List[float] = []
    ppo_no_penalty_route_mean_hit_best_rate: List[float] = []
    bandit_risk_route_mean_hit_best_rate: List[float] = []
    bandit_no_penalty_route_mean_hit_best_rate: List[float] = []

    # best ratio
    ppo_risk_route_mean_best_ratio: List[float] = []
    ppo_no_penalty_route_mean_best_ratio: List[float] = []
    bandit_risk_route_mean_best_ratio: List[float] = []
    bandit_no_penalty_route_mean_best_ratio: List[float] = []
    
    # z
    ppo_risk_route_mean_z: List[float] = []
    ppo_no_penalty_route_mean_z: List[float] = []
    bandit_risk_route_mean_z: List[float] = []
    bandit_no_penalty_route_mean_z: List[float] = []

    # pair ratio
    ppo_risk_route_mean_pair_ratio: List[float] = []
    ppo_no_penalty_route_mean_pair_ratio: List[float] = []
    bandit_risk_route_mean_pair_ratio: List[float] = []
    bandit_no_penalty_route_mean_pair_ratio: List[float] = []

    for route_id in common_ids:
        ppo_risk_path = ppo_risk_map[route_id]
        ppo_no_penalty_path = ppo_no_penalty_map[route_id]
        bandit_risk_path = bandit_risk_map[route_id]
        bandit_no_penalty_path = bandit_no_penalty_map[route_id]
        gt_path = gt_map[route_id]


        # -------------------------
        # chosen rate
        # -------------------------
        ppo_risk_chosen = load_chosen_rate_array(ppo_risk_path)
        ppo_no_penalty_chosen = load_chosen_rate_array(ppo_no_penalty_path)
        bandit_risk_chosen = load_chosen_rate_array(bandit_risk_path)
        bandit_no_penalty_chosen = load_chosen_rate_array(bandit_no_penalty_path)
        
        # -------------------------
        # z
        # -------------------------
        ppo_risk_z = load_z_array(ppo_risk_path)
        ppo_no_penalty_z = load_z_array(ppo_no_penalty_path)
        bandit_risk_z = load_z_array(bandit_risk_path)
        bandit_no_penalty_z = load_z_array(bandit_no_penalty_path)
        
        # -------------------------
        # benchmark reward (recomputed uniformly)
        # -------------------------
        ppo_risk_reward = compute_benchmark_reward_array(
            chosen_rate=ppo_risk_chosen,
            z=ppo_risk_z,
            lambda1=lambda1,
            lambda2=lambda2,
            reward_mode=reward_mode,
        )

        ppo_no_penalty_reward = compute_benchmark_reward_array(
            chosen_rate=ppo_no_penalty_chosen,
            z=ppo_no_penalty_z,
            lambda1=lambda1,
            lambda2=lambda2,
            reward_mode=reward_mode,
        )

        bandit_risk_reward = compute_benchmark_reward_array(
            chosen_rate=bandit_risk_chosen,
            z=bandit_risk_z,
            lambda1=lambda1,
            lambda2=lambda2,
            reward_mode=reward_mode,
        )

        bandit_no_penalty_reward = compute_benchmark_reward_array(
            chosen_rate=bandit_no_penalty_chosen,
            z=bandit_no_penalty_z,
            lambda1=lambda1,
            lambda2=lambda2,
            reward_mode=reward_mode,
        )

        # -------------------------
        # pair ratio
        # -------------------------
        ppo_risk_pair = load_pair_ratio_scalar(ppo_risk_path)
        ppo_no_penalty_pair = load_pair_ratio_scalar(ppo_no_penalty_path)
        bandit_risk_pair = load_pair_ratio_scalar(bandit_risk_path)
        bandit_no_penalty_pair = load_pair_ratio_scalar(bandit_no_penalty_path)
        
        ppo_risk_route_mean_pair_ratio.append(ppo_risk_pair)
        ppo_no_penalty_route_mean_pair_ratio.append(ppo_no_penalty_pair)
        bandit_risk_route_mean_pair_ratio.append(bandit_risk_pair)
        bandit_no_penalty_route_mean_pair_ratio.append(bandit_no_penalty_pair)
        

        # -------------------------
        # best rate from groundtruth
        # -------------------------
        best_rate = load_groundtruth_best_rate(gt_path)
        optimal_reward = np.log1p(best_rate)

        validate_same_length(route_id, ppo_risk_reward, best_rate, "ppo_risk_reward", "best_rate")
        validate_same_length(route_id, ppo_no_penalty_reward, best_rate, "ppo_no_penalty_reward", "best_rate")
        validate_same_length(route_id, bandit_risk_reward, best_rate, "bandit_risk_reward", "best_rate")
        validate_same_length(route_id, bandit_no_penalty_reward, best_rate, "bandit_no_penalty_reward", "best_rate")

        validate_same_length(route_id, ppo_risk_chosen, best_rate, "ppo_risk_chosen", "best_rate")
        validate_same_length(route_id, ppo_no_penalty_chosen, best_rate, "ppo_no_penalty_chosen", "best_rate")
        validate_same_length(route_id, bandit_risk_chosen, best_rate, "bandit_risk_chosen", "best_rate")
        validate_same_length(route_id, bandit_no_penalty_chosen, best_rate, "bandit_no_penalty_chosen", "best_rate")
        
        validate_same_length(route_id, ppo_risk_z, best_rate, "ppo_risk_z", "best_rate")
        validate_same_length(route_id, ppo_no_penalty_z, best_rate, "ppo_no_penalty_z", "best_rate")
        validate_same_length(route_id, bandit_risk_z, best_rate, "bandit_risk_z", "best_rate")
        validate_same_length(route_id, bandit_no_penalty_z, best_rate, "bandit_no_penalty_z", "best_rate")

        # reward summary
        ppo_risk_route_mean_reward.append(float(np.mean(ppo_risk_reward)))
        ppo_no_penalty_route_mean_reward.append(float(np.mean(ppo_no_penalty_reward)))
        bandit_risk_route_mean_reward.append(float(np.mean(bandit_risk_reward)))
        bandit_no_penalty_route_mean_reward.append(float(np.mean(bandit_no_penalty_reward)))
        optimal_route_mean_reward.append(float(np.mean(optimal_reward)))

        # -------------------------
        # hit best
        # -------------------------
        ppo_risk_hit = compute_hit_best_array(ppo_risk_chosen, best_rate)
        ppo_no_penalty_hit = compute_hit_best_array(ppo_no_penalty_chosen, best_rate)
        bandit_risk_hit = compute_hit_best_array(bandit_risk_chosen, best_rate)
        bandit_no_penalty_hit = compute_hit_best_array(bandit_no_penalty_chosen, best_rate)

        ppo_risk_route_mean_hit_best_rate.append(float(np.mean(ppo_risk_hit)))
        ppo_no_penalty_route_mean_hit_best_rate.append(float(np.mean(ppo_no_penalty_hit)))
        bandit_risk_route_mean_hit_best_rate.append(float(np.mean(bandit_risk_hit)))
        bandit_no_penalty_route_mean_hit_best_rate.append(float(np.mean(bandit_no_penalty_hit)))

        # -------------------------
        # best ratio
        # -------------------------
        ppo_risk_ratio = compute_best_ratio_array(ppo_risk_chosen, best_rate)
        ppo_no_penalty_ratio = compute_best_ratio_array(ppo_no_penalty_chosen, best_rate)
        bandit_risk_ratio = compute_best_ratio_array(bandit_risk_chosen, best_rate)
        bandit_no_penalty_ratio = compute_best_ratio_array(bandit_no_penalty_chosen, best_rate)

        ppo_risk_route_mean_best_ratio.append(float(np.mean(ppo_risk_ratio)))
        ppo_no_penalty_route_mean_best_ratio.append(float(np.mean(ppo_no_penalty_ratio)))
        bandit_risk_route_mean_best_ratio.append(float(np.mean(bandit_risk_ratio)))
        bandit_no_penalty_route_mean_best_ratio.append(float(np.mean(bandit_no_penalty_ratio)))
        
        # z summary
        ppo_risk_route_mean_z.append(float(np.mean(ppo_risk_z)))
        ppo_no_penalty_route_mean_z.append(float(np.mean(ppo_no_penalty_z)))
        bandit_risk_route_mean_z.append(float(np.mean(bandit_risk_z)))
        bandit_no_penalty_route_mean_z.append(float(np.mean(bandit_no_penalty_z)))


    ppo_risk_route_mean_reward = np.asarray(ppo_risk_route_mean_reward, dtype=np.float64)
    ppo_no_penalty_route_mean_reward = np.asarray(ppo_no_penalty_route_mean_reward, dtype=np.float64)
    bandit_risk_route_mean_reward = np.asarray(bandit_risk_route_mean_reward, dtype=np.float64)
    bandit_no_penalty_route_mean_reward = np.asarray(bandit_no_penalty_route_mean_reward, dtype=np.float64)
    optimal_route_mean_reward = np.asarray(optimal_route_mean_reward, dtype=np.float64)

    ppo_risk_route_mean_hit_best_rate = np.asarray(ppo_risk_route_mean_hit_best_rate, dtype=np.float64)
    ppo_no_penalty_route_mean_hit_best_rate = np.asarray(ppo_no_penalty_route_mean_hit_best_rate, dtype=np.float64)
    bandit_risk_route_mean_hit_best_rate = np.asarray(bandit_risk_route_mean_hit_best_rate, dtype=np.float64)
    bandit_no_penalty_route_mean_hit_best_rate = np.asarray(bandit_no_penalty_route_mean_hit_best_rate, dtype=np.float64)

    ppo_risk_route_mean_best_ratio = np.asarray(ppo_risk_route_mean_best_ratio, dtype=np.float64)
    ppo_no_penalty_route_mean_best_ratio = np.asarray(ppo_no_penalty_route_mean_best_ratio, dtype=np.float64)
    bandit_risk_route_mean_best_ratio = np.asarray(bandit_risk_route_mean_best_ratio, dtype=np.float64)
    bandit_no_penalty_route_mean_best_ratio = np.asarray(bandit_no_penalty_route_mean_best_ratio, dtype=np.float64)
    
    ppo_risk_route_mean_z = np.asarray(ppo_risk_route_mean_z, dtype=np.float64)
    ppo_no_penalty_route_mean_z = np.asarray(ppo_no_penalty_route_mean_z, dtype=np.float64)
    bandit_risk_route_mean_z = np.asarray(bandit_risk_route_mean_z, dtype=np.float64)
    bandit_no_penalty_route_mean_z = np.asarray(bandit_no_penalty_route_mean_z, dtype=np.float64)

    ppo_risk_route_mean_pair_ratio = np.asarray(ppo_risk_route_mean_pair_ratio, dtype=np.float64)
    ppo_no_penalty_route_mean_pair_ratio = np.asarray(ppo_no_penalty_route_mean_pair_ratio, dtype=np.float64)
    bandit_risk_route_mean_pair_ratio = np.asarray(bandit_risk_route_mean_pair_ratio, dtype=np.float64)
    bandit_no_penalty_route_mean_pair_ratio = np.asarray(bandit_no_penalty_route_mean_pair_ratio, dtype=np.float64)

    # ------------------------------------------------------------
    # Plot 1: reward
    # ------------------------------------------------------------
    plot_CCDF_multi(
        values_dict={
            "PPO (risk-aware)": ppo_risk_route_mean_reward,
            "PPO": ppo_no_penalty_route_mean_reward,
            "Bandit (risk-aware)": bandit_risk_route_mean_reward,
            "Bandit": bandit_no_penalty_route_mean_reward,
            "Optimal": optimal_route_mean_reward,
        },
        xlabel="Route mean reward",
        title=(
            f"Route-level Mean Reward CCDF\n"
            # f"(risk-aware vs no-penalty)\n"
            f"(λ1={lambda1}, λ2={lambda2}, ε={epsilon_budget}, α={alpha_risk})"
        ),
        save_path=OUTPUT_DIR / "reward_CCDF_route_overlay.png",
    )

    # ------------------------------------------------------------
    # Plot 2: hit best rate
    # ------------------------------------------------------------
    plot_CCDF_multi(
        values_dict={
            "PPO (risk-aware)": ppo_risk_route_mean_hit_best_rate,
            "PPO": ppo_no_penalty_route_mean_hit_best_rate,
            "Bandit (risk-aware)": bandit_risk_route_mean_hit_best_rate,
            "Bandit": bandit_no_penalty_route_mean_hit_best_rate,
        },
        xlabel="Route mean hit-best rate",
        title=(
            f"Route-level Mean Hit-Best-Rate CCDF\n"
            f"(λ1={lambda1}, λ2={lambda2}, ε={epsilon_budget}, α={alpha_risk})"
        ),
        save_path=OUTPUT_DIR / "hit_best_rate_CCDF_route_overlay.png",
    )

    # ------------------------------------------------------------
    # Plot 3: best ratio
    # ------------------------------------------------------------
    plot_CCDF_multi(
        values_dict={
            "PPO (risk-aware)": ppo_risk_route_mean_best_ratio,
            "PPO": ppo_no_penalty_route_mean_best_ratio,
            "Bandit (risk-aware)": bandit_risk_route_mean_best_ratio,
            "Bandit": bandit_no_penalty_route_mean_best_ratio,
        },
        xlabel="Route mean best ratio",
        title=(
            f"Route-level Mean Best-Ratio CCDF\n"
            f"(λ1={lambda1}, λ2={lambda2}, ε={epsilon_budget}, α={alpha_risk})"
        ),
        save_path=OUTPUT_DIR / "best_ratio_CCDF_route_overlay.png",
    )
    
    # ------------------------------------------------------------
    # Plot 4: z
    # ------------------------------------------------------------
    plot_CCDF_multi(
        values_dict={
            "PPO (risk-aware)": ppo_risk_route_mean_z,
            "PPO": ppo_no_penalty_route_mean_z,
            # "Bandit (risk-aware)": bandit_risk_route_mean_z,
            # "Bandit (no penalty)": bandit_no_penalty_route_mean_z,
            "Bandit": bandit_no_penalty_route_mean_z,
        },
        xlabel="Route mean z",
        title=(
            f"Route-level Mean z CCDF\n"
            f"(λ1={lambda1}, λ2={lambda2}, ε={epsilon_budget}, α={alpha_risk})"
        ),
        save_path=OUTPUT_DIR / "z_CCDF_route_overlay.png",
    )

    # ------------------------------------------------------------
    # Plot 5: pair ratio
    # ------------------------------------------------------------
    plot_CCDF_multi(
        values_dict={
            "PPO (risk-aware)": ppo_risk_route_mean_pair_ratio,
            "PPO": ppo_no_penalty_route_mean_pair_ratio,
            # "Bandit (risk-aware)": bandit_risk_route_mean_pair_ratio,
            # "Bandit (no penalty)": bandit_no_penalty_route_mean_pair_ratio,
            "Bandit": bandit_no_penalty_route_mean_pair_ratio,
        },
        xlabel="Route mean pair ratio",
        title=(
            f"Route-level Mean Pair-Ratio CCDF\n"
            f"(λ1={lambda1}, λ2={lambda2}, ε={epsilon_budget}, α={alpha_risk})"
        ),
        save_path=OUTPUT_DIR / "pair_ratio_CCDF_route_overlay.png",
    )

    summary = {
        "num_routes_matched": int(len(common_ids)),

        "ppo_risk_route_mean_reward_mean": float(np.mean(ppo_risk_route_mean_reward)),
        "ppo_no_penalty_route_mean_reward_mean": float(np.mean(ppo_no_penalty_route_mean_reward)),
        "bandit_risk_route_mean_reward_mean": float(np.mean(bandit_risk_route_mean_reward)),
        "bandit_no_penalty_route_mean_reward_mean": float(np.mean(bandit_no_penalty_route_mean_reward)),
        "optimal_route_mean_reward_mean": float(np.mean(optimal_route_mean_reward)),

        "ppo_risk_route_mean_reward_median": float(np.median(ppo_risk_route_mean_reward)),
        "ppo_no_penalty_route_mean_reward_median": float(np.median(ppo_no_penalty_route_mean_reward)),
        "bandit_risk_route_mean_reward_median": float(np.median(bandit_risk_route_mean_reward)),
        "bandit_no_penalty_route_mean_reward_median": float(np.median(bandit_no_penalty_route_mean_reward)),
        "optimal_route_mean_reward_median": float(np.median(optimal_route_mean_reward)),

        "ppo_risk_route_mean_hit_best_rate_mean": float(np.mean(ppo_risk_route_mean_hit_best_rate)),
        "ppo_no_penalty_route_mean_hit_best_rate_mean": float(np.mean(ppo_no_penalty_route_mean_hit_best_rate)),
        "bandit_risk_route_mean_hit_best_rate_mean": float(np.mean(bandit_risk_route_mean_hit_best_rate)),
        "bandit_no_penalty_route_mean_hit_best_rate_mean": float(np.mean(bandit_no_penalty_route_mean_hit_best_rate)),

        "ppo_risk_route_mean_hit_best_rate_median": float(np.median(ppo_risk_route_mean_hit_best_rate)),
        "ppo_no_penalty_route_mean_hit_best_rate_median": float(np.median(ppo_no_penalty_route_mean_hit_best_rate)),
        "bandit_risk_route_mean_hit_best_rate_median": float(np.median(bandit_risk_route_mean_hit_best_rate)),
        "bandit_no_penalty_route_mean_hit_best_rate_median": float(np.median(bandit_no_penalty_route_mean_hit_best_rate)),

        "ppo_risk_route_mean_best_ratio_mean": float(np.mean(ppo_risk_route_mean_best_ratio)),
        "ppo_no_penalty_route_mean_best_ratio_mean": float(np.mean(ppo_no_penalty_route_mean_best_ratio)),
        "bandit_risk_route_mean_best_ratio_mean": float(np.mean(bandit_risk_route_mean_best_ratio)),
        "bandit_no_penalty_route_mean_best_ratio_mean": float(np.mean(bandit_no_penalty_route_mean_best_ratio)),

        "ppo_risk_route_mean_best_ratio_median": float(np.median(ppo_risk_route_mean_best_ratio)),
        "ppo_no_penalty_route_mean_best_ratio_median": float(np.median(ppo_no_penalty_route_mean_best_ratio)),
        "bandit_risk_route_mean_best_ratio_median": float(np.median(bandit_risk_route_mean_best_ratio)),
        "bandit_no_penalty_route_mean_best_ratio_median": float(np.median(bandit_no_penalty_route_mean_best_ratio)),
        
        "ppo_risk_route_mean_z_mean": float(np.mean(ppo_risk_route_mean_z)),
        "ppo_no_penalty_route_mean_z_mean": float(np.mean(ppo_no_penalty_route_mean_z)),
        "bandit_risk_route_mean_z_mean": float(np.mean(bandit_risk_route_mean_z)),
        "bandit_no_penalty_route_mean_z_mean": float(np.mean(bandit_no_penalty_route_mean_z)),

        "ppo_risk_route_mean_z_median": float(np.median(ppo_risk_route_mean_z)),
        "ppo_no_penalty_route_mean_z_median": float(np.median(ppo_no_penalty_route_mean_z)),
        "bandit_risk_route_mean_z_median": float(np.median(bandit_risk_route_mean_z)),
        "bandit_no_penalty_route_mean_z_median": float(np.median(bandit_no_penalty_route_mean_z)),

        "ppo_risk_route_mean_pair_ratio_mean": float(np.mean(ppo_risk_route_mean_pair_ratio)),
        "ppo_no_penalty_route_mean_pair_ratio_mean": float(np.mean(ppo_no_penalty_route_mean_pair_ratio)),
        "bandit_risk_route_mean_pair_ratio_mean": float(np.mean(bandit_risk_route_mean_pair_ratio)),
        "bandit_no_penalty_route_mean_pair_ratio_mean": float(np.mean(bandit_no_penalty_route_mean_pair_ratio)),

        "ppo_risk_route_mean_pair_ratio_median": float(np.median(ppo_risk_route_mean_pair_ratio)),
        "ppo_no_penalty_route_mean_pair_ratio_median": float(np.median(ppo_no_penalty_route_mean_pair_ratio)),
        "bandit_risk_route_mean_pair_ratio_median": float(np.median(bandit_risk_route_mean_pair_ratio)),
        "bandit_no_penalty_route_mean_pair_ratio_median": float(np.median(bandit_no_penalty_route_mean_pair_ratio)),
    }

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=== DONE ===")
    logger.info(json.dumps(summary, indent=2))
    logger.info(f"Saved: {OUTPUT_DIR / 'reward_CCDF_route_overlay.png'}")
    logger.info(f"Saved: {OUTPUT_DIR / 'hit_best_rate_CCDF_route_overlay.png'}")
    logger.info(f"Saved: {OUTPUT_DIR / 'best_ratio_CCDF_route_overlay.png'}")
    logger.info(f"Saved: {OUTPUT_DIR / 'z_CCDF_route_overlay.png'}")
    logger.info(f"Saved: {OUTPUT_DIR / 'pair_ratio_CCDF_route_overlay.png'}")


if __name__ == "__main__":
    main()