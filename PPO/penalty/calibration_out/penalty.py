"""
Penalty calibration for multi-array exploration cost.

This version uses an anchor-extra definition:

    extra_gain(a -> b, t) = best_rate({a,b}, t) - best_rate({a}, t)

Meaning:
    if array a is already active, how much extra rate can be gained
    by additionally activating array b?

This is usually more meaningful than comparing:
    best({a,b}) - max(best(a), best(b))
which is identically zero in this observation model.

Usage:
    1) Edit parameters in main()
    2) Run:
           python penalty.py
    3) Reload later without recomputing:
           res = PenaltyCalibrationResult.load("calibration_out")
           print(res.suggest_penalty("median_ratio", ratio=0.5))
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import json

import numpy as np


# =========================================================
# Dataset loading
# =========================================================

def load_route_capacity(npz_path: Path) -> np.ndarray:
    """
    Load one route file and convert capacity to shape [T, N_cell, N_arr].

    Expected raw shape from your dataset after squeeze:
        [T, N_arr, N_cell]

    Returns
    -------
    np.ndarray
        Rate tensor with shape [T, N_cell, N_arr], dtype float.
    """
    data = np.load(npz_path, allow_pickle=True)

    if "capacity" not in data:
        raise KeyError(f"'capacity' not found in {npz_path}")

    cap = data["capacity"]

    # Remove singleton dimensions.
    cap = np.squeeze(np.array(cap))

    # Convert object dtype to float.
    cap = cap.astype(float)

    if cap.ndim != 3:
        raise ValueError(
            f"Expected squeezed capacity to have 3 dims, got shape {cap.shape} in {npz_path}"
        )

    # Your dataset is [T, N_arr, N_cell].
    # Convert to the internal convention [T, N_cell, N_arr].
    cap = cap.transpose(0, 2, 1)

    return cap


def load_dataset_folder(folder: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load all *_result.npz files from a folder and concatenate along time.

    Returns
    -------
    R : np.ndarray
        Shape [T_total, N_cell, N_arr]
    route_names : list[str]
        File names in loading order.
    """
    folder_path = Path(folder)
    files = sorted(folder_path.glob("*_result.npz"))

    if not files:
        raise RuntimeError(f"No *_result.npz files found in: {folder_path}")

    all_caps: List[np.ndarray] = []
    route_names: List[str] = []

    ref_shape = None

    for file_path in files:
        cap = load_route_capacity(file_path)

        if ref_shape is None:
            ref_shape = cap.shape[1:]  # (N_cell, N_arr)
        elif cap.shape[1:] != ref_shape:
            raise ValueError(
                f"Inconsistent per-route shape: expected (*, {ref_shape[0]}, {ref_shape[1]}), "
                f"got {cap.shape} from {file_path}"
            )

        all_caps.append(cap)
        route_names.append(file_path.name)

    R = np.concatenate(all_caps, axis=0)
    return R, route_names


# =========================================================
# Subset / action utilities
# =========================================================

def generate_single_and_pair_subsets(n_arr: int) -> Tuple[List[Tuple[int]], List[Tuple[int, int]]]:
    """
    Generate all size-1 and size-2 array subsets.
    """
    single_subsets = [(a,) for a in range(n_arr)]
    pair_subsets = [(a, b) for a in range(n_arr) for b in range(a + 1, n_arr)]
    return single_subsets, pair_subsets


def generate_anchor_extra_actions(n_arr: int) -> List[Tuple[int, int]]:
    """
    Generate all ordered anchor->extra actions with anchor != extra.

    Example for n_arr=3:
        (0,1), (0,2), (1,0), (1,2), (2,0), (2,1)
    """
    actions = []
    for anchor in range(n_arr):
        for extra in range(n_arr):
            if extra != anchor:
                actions.append((anchor, extra))
    return actions


# =========================================================
# Core best-rate computations
# =========================================================

def best_rate_for_subset(R: np.ndarray, subset: Tuple[int, ...]) -> np.ndarray:
    """
    Compute best observable rate for one subset over time.

    Parameters
    ----------
    R : np.ndarray
        Shape [T, N_cell, N_arr]
    subset : tuple[int, ...]
        Selected arrays

    Returns
    -------
    np.ndarray
        Shape [T], best feasible rate at each time.
    """
    if R.ndim != 3:
        raise ValueError(f"R must be 3D [T, N_cell, N_arr], got shape {R.shape}")

    sub = R[:, :, subset]  # [T, N_cell, |subset|]

    # If all values are NaN at some time, nanmax returns NaN.
    return np.nanmax(sub, axis=(1, 2))


def compute_single_and_pair_rates(
    R: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int]], List[Tuple[int, int]]]:
    """
    Precompute best rates for all size-1 and size-2 subsets.

    Returns
    -------
    single_rates : np.ndarray
        Shape [T, N_arr]
    pair_rates : np.ndarray
        Shape [T, N_pair]
    single_subsets : list[tuple[int]]
    pair_subsets : list[tuple[int, int]]
    """
    _, _, n_arr = R.shape
    single_subsets, pair_subsets = generate_single_and_pair_subsets(n_arr)

    single_rates = np.stack(
        [best_rate_for_subset(R, s) for s in single_subsets],
        axis=1
    )  # [T, N_arr]

    pair_rates = np.stack(
        [best_rate_for_subset(R, p) for p in pair_subsets],
        axis=1
    )  # [T, N_pair]

    return single_rates, pair_rates, single_subsets, pair_subsets


# =========================================================
# Statistics
# =========================================================

def safe_scalar_stat(arr: np.ndarray, fn_name: str, **kwargs: Any) -> float:
    """
    Return a scalar float statistic over finite values only.
    """
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan")

    if fn_name == "mean":
        return float(np.mean(finite))
    if fn_name == "median":
        return float(np.median(finite))
    if fn_name == "quantile":
        q = kwargs["q"]
        return float(np.quantile(finite, q))

    raise ValueError(f"Unsupported fn_name: {fn_name}")


def compute_statistics(R: np.ndarray) -> Dict[str, Any]:
    """
    Compute offline statistics for penalty calibration.

    Main recommended metric:
        extra_gain(anchor -> extra, t) = best({anchor, extra}) - best({anchor})

    Returns
    -------
    dict
        Statistics and precomputed arrays.
    """
    if R.ndim != 3:
        raise ValueError(f"R must be 3D [T, N_cell, N_arr], got shape {R.shape}")

    T, N_cell, N_arr = R.shape

    single_rates, pair_rates, single_subsets, pair_subsets = compute_single_and_pair_rates(R)

    # Mostly for debugging / sanity checking.
    best_single_array_rate = np.nanmax(single_rates, axis=1)
    best_two_array_rate = np.nanmax(pair_rates, axis=1)
    delta_best_global = best_two_array_rate - best_single_array_rate

    # Map unordered pair -> pair_rates column index.
    pair_to_index = {pair: idx for idx, pair in enumerate(pair_subsets)}

    anchor_extra_actions = generate_anchor_extra_actions(N_arr)

    # Ordered gains: [T, N_anchor_extra_actions]
    anchor_extra_gains_list = []
    anchor_extra_labels = []

    for anchor, extra in anchor_extra_actions:
        pair = (min(anchor, extra), max(anchor, extra))
        pair_idx = pair_to_index[pair]

        best_anchor = single_rates[:, anchor]
        best_pair = pair_rates[:, pair_idx]

        gain = best_pair - best_anchor
        anchor_extra_gains_list.append(gain)
        anchor_extra_labels.append(f"{anchor}->{extra}")

    anchor_extra_gains = np.stack(anchor_extra_gains_list, axis=1)  # [T, N_ordered_actions]
    anchor_extra_gains_flat = anchor_extra_gains.reshape(-1)

    # Per-action average gains.
    mean_gain_by_action = np.array(
        [safe_scalar_stat(anchor_extra_gains[:, i], "mean") for i in range(anchor_extra_gains.shape[1])]
    )
    median_gain_by_action = np.array(
        [safe_scalar_stat(anchor_extra_gains[:, i], "median") for i in range(anchor_extra_gains.shape[1])]
    )

    stats: Dict[str, Any] = {
        # Metadata
        "T_total": int(T),
        "n_cell": int(N_cell),
        "n_arr": int(N_arr),
        "single_subsets": single_subsets,
        "pair_subsets": pair_subsets,
        "anchor_extra_actions": anchor_extra_actions,
        "anchor_extra_labels": anchor_extra_labels,

        # Core arrays
        "single_rates": single_rates,                      # [T, N_arr]
        "pair_rates": pair_rates,                          # [T, N_pair]
        "best_single_array_rate": best_single_array_rate,  # [T]
        "best_two_array_rate": best_two_array_rate,        # [T]
        "delta_best_global": delta_best_global,            # [T]

        # Recommended metric
        "anchor_extra_gains": anchor_extra_gains,          # [T, N_ordered_actions]
        "anchor_extra_gains_flat": anchor_extra_gains_flat,# [T * N_ordered_actions]

        # Summary stats over all ordered anchor->extra gains
        "mean_gain": safe_scalar_stat(anchor_extra_gains_flat, "mean"),
        "median_gain": safe_scalar_stat(anchor_extra_gains_flat, "median"),
        "q75_gain": safe_scalar_stat(anchor_extra_gains_flat, "quantile", q=0.75),
        "q90_gain": safe_scalar_stat(anchor_extra_gains_flat, "quantile", q=0.90),
        "q95_gain": safe_scalar_stat(anchor_extra_gains_flat, "quantile", q=0.95),

        # Per-action summaries
        "mean_gain_by_action": mean_gain_by_action,
        "median_gain_by_action": median_gain_by_action,
    }

    return stats


# =========================================================
# Save / load
# =========================================================

def _json_convert(obj: Any) -> Any:
    """
    Convert Python objects into JSON-serializable forms.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, list):
        return [_json_convert(x) for x in obj]
    return obj


class PenaltyCalibrationResult:
    """
    Lightweight wrapper for saved penalty calibration results.
    """

    def __init__(self, stats: Dict[str, Any], path: Path):
        self.stats = stats
        self.path = path

    @staticmethod
    def save(stats: Dict[str, Any], output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Save ndarray entries to NPZ.
        array_items = {k: v for k, v in stats.items() if isinstance(v, np.ndarray)}
        np.savez_compressed(out / "stats.npz", **array_items)

        # Save lightweight summary / metadata to JSON.
        summary_items = {k: v for k, v in stats.items() if not isinstance(v, np.ndarray)}
        summary_items = {k: _json_convert(v) for k, v in summary_items.items()}

        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary_items, f, indent=2)

    @staticmethod
    def load(output_dir: str) -> "PenaltyCalibrationResult":
        out = Path(output_dir)

        npz_path = out / "stats.npz"
        json_path = out / "summary.json"

        if not npz_path.exists():
            raise FileNotFoundError(f"Missing file: {npz_path}")
        if not json_path.exists():
            raise FileNotFoundError(f"Missing file: {json_path}")

        arrays = dict(np.load(npz_path, allow_pickle=True))
        with open(json_path, "r", encoding="utf-8") as f:
            summary = json.load(f)

        stats = {**summary, **arrays}
        return PenaltyCalibrationResult(stats, out)

    def suggest_penalty(
        self,
        method: str = "median_ratio",
        ratio: float = 1.0,
        q: float = 0.9,
    ) -> float:
        """
        Suggest lambda_E from saved gain statistics.

        Supported methods
        -----------------
        mean_ratio:
            lambda_E = ratio * mean(anchor_extra_gain)

        median_ratio:
            lambda_E = ratio * median(anchor_extra_gain)

        quantile_ratio:
            lambda_E = ratio * quantile(anchor_extra_gain, q)

        fixed:
            lambda_E = ratio
        """
        gains = np.asarray(self.stats["anchor_extra_gains_flat"], dtype=float)
        finite = gains[np.isfinite(gains)]

        if finite.size == 0:
            return float("nan")

        if method == "mean_ratio":
            base = np.mean(finite)
        elif method == "median_ratio":
            base = np.median(finite)
        elif method == "quantile_ratio":
            base = np.quantile(finite, q)
        elif method == "fixed":
            return float(ratio)
        else:
            raise ValueError(f"Unknown method: {method}")

        return float(ratio * base)


# =========================================================
# Optional reporting
# =========================================================

def print_summary(stats: Dict[str, Any]) -> None:
    """
    Print a concise human-readable summary.
    """
    print("\n=== Dataset summary ===")
    print(f"T_total : {stats['T_total']}")
    print(f"N_cell  : {stats['n_cell']}")
    print(f"N_arr   : {stats['n_arr']}")

    print("\n=== Global best sanity check ===")
    print(f"mean(delta_best_global):   {safe_scalar_stat(stats['delta_best_global'], 'mean'):.6f}")
    print(f"median(delta_best_global): {safe_scalar_stat(stats['delta_best_global'], 'median'):.6f}")

    print("\n=== Anchor-extra gain summary ===")
    print(f"mean_gain   : {stats['mean_gain']:.6f}")
    print(f"median_gain : {stats['median_gain']:.6f}")
    print(f"q75_gain    : {stats['q75_gain']:.6f}")
    print(f"q90_gain    : {stats['q90_gain']:.6f}")
    print(f"q95_gain    : {stats['q95_gain']:.6f}")

    print("\n=== Mean gain by ordered action ===")
    labels = stats["anchor_extra_labels"]
    values = stats["mean_gain_by_action"]
    for label, value in zip(labels, values):
        print(f"{label:>6s} : {value:.6f}")


# =========================================================
# MAIN
# =========================================================

def main() -> None:
    # -----------------------------------------------------
    # Edit parameters here
    # -----------------------------------------------------
    dataset_dir = "../dataset/CQI_Mobility_medium_R_2_region_2"
    output_dir = "./calibration_out"

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------
    print("Loading dataset...")
    R, route_names = load_dataset_folder(dataset_dir)
    print("R shape:", R.shape)
    print("Number of route files:", len(route_names))

    # -----------------------------------------------------
    # Compute and save once
    # -----------------------------------------------------
    print("Computing statistics...")
    stats = compute_statistics(R)

    print("Saving...")
    PenaltyCalibrationResult.save(stats, output_dir)

    print("Done.")

    # -----------------------------------------------------
    # Reload and suggest penalties without recomputing
    # -----------------------------------------------------
    res = PenaltyCalibrationResult.load(output_dir)

    print_summary(res.stats)

    print("\n=== Suggested lambda_E ===")
    print("mean   x0.5 :", res.suggest_penalty(method="mean_ratio", ratio=0.5))
    print("median x0.5 :", res.suggest_penalty(method="median_ratio", ratio=0.5))
    print("q90    x1.0 :", res.suggest_penalty(method="quantile_ratio", ratio=1.0, q=0.90))
    print("q95    x1.0 :", res.suggest_penalty(method="quantile_ratio", ratio=1.0, q=0.95))


if __name__ == "__main__":
    main()