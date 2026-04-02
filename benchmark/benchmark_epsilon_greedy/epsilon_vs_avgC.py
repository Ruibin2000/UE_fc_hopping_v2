from pathlib import Path
from datetime import datetime
import os
import json
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


###########################################################################
# Core algorithm
###########################################################################
def epsilon_single_ue(capacity, epsilon=0.3, seed=None, verbose=False):
    """
    Sticky RX-level exploration for a single UE.

    Parameters
    ----------
    capacity : np.ndarray
        Shape (N_sample, N_RX, N_TX)
    epsilon : float
        Exploration probability.
    seed : int or None
        Random seed.
    verbose : bool
        If True, print detailed link-capacity comparison at each exploration step.

    Returns
    -------
    dict
        {
            "serving_rx":       (N_sample,),
            "serving_tx":       (N_sample,),
            "serving_capacity": (N_sample,),
            "explore_flag":     (N_sample,),
            "switch_flag":      (N_sample,),
            "new_rx":           (N_sample,),   # np.nan if no exploration
        }
    """
    capacity = np.asarray(capacity)
    if capacity.ndim != 3:
        raise ValueError("capacity must have shape (N_sample, N_RX, N_TX)")

    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")

    n_sample, n_rx, n_tx = capacity.shape
    rng = np.random.default_rng(seed)

    serving_rx = np.full(n_sample, -1, dtype=int)
    serving_tx = np.full(n_sample, -1, dtype=int)
    serving_capacity = np.zeros(n_sample, dtype=capacity.dtype)

    explore_flag = np.zeros(n_sample, dtype=bool)
    switch_flag = np.zeros(n_sample, dtype=bool)
    new_rx_record = np.full(n_sample, np.nan, dtype=float)

    # t = 0: choose global best link
    mat0 = capacity[0]  # (N_RX, N_TX)
    flat_idx = np.argmax(mat0)
    rx0, tx0 = np.unravel_index(flat_idx, mat0.shape)

    serving_rx[0] = rx0
    serving_tx[0] = tx0
    serving_capacity[0] = mat0[rx0, tx0]

    if verbose:
        print(f"[t=0] INIT")
        print(f"  selected: RX={rx0}, TX={tx0}, cap={serving_capacity[0]:.4f}")

    # t >= 1
    for t in range(1, n_sample):
        prev_rx = serving_rx[t - 1]
        prev_tx = serving_tx[t - 1]

        # keep previous serving link
        if rng.random() <= (1.0 - epsilon):
            serving_rx[t] = prev_rx
            serving_tx[t] = prev_tx
            serving_capacity[t] = capacity[t, prev_rx, prev_tx]

        # explore one new RX and compare
        else:
            explore_flag[t] = True

            candidate_rxs = [rx for rx in range(n_rx) if rx != prev_rx]
            rx_new = int(rng.choice(candidate_rxs))
            new_rx_record[t] = float(rx_new)

            rx_pair = [prev_rx, rx_new]
            submat = capacity[t, rx_pair, :]  # (2, N_TX)

            flat_idx = np.argmax(submat)
            local_rx_idx, best_tx = np.unravel_index(flat_idx, submat.shape)

            best_rx = rx_pair[local_rx_idx]
            best_cap = submat[local_rx_idx, best_tx]

            serving_rx[t] = best_rx
            serving_tx[t] = best_tx
            serving_capacity[t] = best_cap

            switched = (best_rx != prev_rx)
            switch_flag[t] = switched

            if verbose:
                print(f"\n[t={t}] EXPLORE")
                print(
                    f"  prev serving: RX={prev_rx}, TX={prev_tx}, "
                    f"cap_now={capacity[t, prev_rx, prev_tx]:.4f}"
                )
                print(f"  new_rx tried: RX={rx_new}")

                for i, rx in enumerate(rx_pair):
                    caps = submat[i]
                    best_tx_i = int(np.argmax(caps))
                    best_cap_i = caps[best_tx_i]
                    tag = "OLD" if rx == prev_rx else "NEW"
                    print(f"  RX {rx} ({tag}) all TX caps: {caps}")
                    print(f"      best on RX {rx}: TX={best_tx_i}, cap={best_cap_i:.4f}")

                print(f"  ==> selected: RX={best_rx}, TX={best_tx}, cap={best_cap:.4f}")
                print(f"  ==> switched RX: {switched}")

    return {
        "serving_rx": serving_rx,
        "serving_tx": serving_tx,
        "serving_capacity": serving_capacity,
        "explore_flag": explore_flag,
        "switch_flag": switch_flag,
        "new_rx": new_rx_record,
    }


###########################################################################
# Metrics
###########################################################################
def summarize_selection_performance(capacity, res):
    """
    Summarize selection performance for one route.

    Parameters
    ----------
    capacity : np.ndarray
        Shape (N_sample, N_RX, N_TX)
    res : dict
        Output of epsilon_single_ue(...)

    Returns
    -------
    dict
        Route-level statistics.
    """
    capacity = np.asarray(capacity)

    serving_rx = np.asarray(res["serving_rx"])
    serving_tx = np.asarray(res["serving_tx"])
    serving_capacity = np.asarray(res["serving_capacity"])
    switch_flag = np.asarray(res["switch_flag"]).astype(bool)

    n_sample, n_rx, n_tx = capacity.shape

    # global best capacity at each step
    best_capacity = np.max(capacity, axis=(1, 2))

    # global best link at each step
    best_flat_idx = np.argmax(capacity.reshape(n_sample, -1), axis=1)
    best_rx, best_tx = np.unravel_index(best_flat_idx, (n_rx, n_tx))

    selected_best_flag = (serving_rx == best_rx) & (serving_tx == best_tx)

    stats = {
        "n_samples": int(n_sample),
        "best_selection_percentage": float(100.0 * np.mean(selected_best_flag)),
        "num_switches": int(np.sum(switch_flag)),
        "avg_best_capacity": float(np.mean(best_capacity)),
        "avg_selected_capacity": float(np.mean(serving_capacity)),
        "avg_gap": float(np.mean(best_capacity - serving_capacity)),
        "max_gap": float(np.max(best_capacity - serving_capacity)),
    }
    return stats


###########################################################################
# Data loading
###########################################################################
def load_channel(npz_path):
    data = np.load(npz_path, allow_pickle=True)

    cap = data["capacity"]
    sinr_db = data["sinr_db"]

    # object array -> stack
    if cap.dtype == object:
        cap = np.stack(cap, axis=0)
    cap = np.squeeze(cap, axis=(3, 5, 6))

    if sinr_db.dtype == object:
        sinr_db = np.stack(sinr_db, axis=0)
    sinr_db = np.squeeze(sinr_db, axis=(3, 5, 6))

    return cap, sinr_db


###########################################################################
# Dataset-level aggregation
###########################################################################
def evaluate_epsilon_on_dataset(npz_files, epsilon, base_seed=42, logger=None):
    """
    Run one epsilon over the whole dataset and only keep epsilon-level summary.

    Returns
    -------
    dict
        Overall statistics aggregated over all routes and all time samples.
    """
    total_samples = 0
    total_switches = 0
    total_selected_capacity_sum = 0.0
    total_best_capacity_sum = 0.0
    total_selected_best_count = 0.0
    total_routes = 0

    for route_idx, npz_path in enumerate(
        tqdm(npz_files, desc=f"epsilon={epsilon:.4f}", ncols=100)
    ):
        capacity, sinr_db = load_channel(npz_path)

        # keep only one UE dimension as in your current code
        capacity = capacity[:, 0, :]   # -> (N_sample, N_RX, N_TX)
        _ = sinr_db[:, 0, :]           # not used now, keep line for consistency

        res = epsilon_single_ue(
            capacity=capacity,
            epsilon=epsilon,
            seed=base_seed + route_idx,
            verbose=False,
        )

        stats = summarize_selection_performance(capacity, res)

        n = stats["n_samples"]
        total_samples += n
        total_switches += stats["num_switches"]
        total_selected_capacity_sum += stats["avg_selected_capacity"] * n
        total_best_capacity_sum += stats["avg_best_capacity"] * n
        total_selected_best_count += (stats["best_selection_percentage"] / 100.0) * n
        total_routes += 1

    if total_samples == 0:
        raise RuntimeError("No samples found when evaluating epsilon.")

    overall = {
        "epsilon": float(epsilon),
        "inv_epsilon": float(1.0 / epsilon) if epsilon > 0 else np.inf,
        "avg_selected_capacity": float(total_selected_capacity_sum / total_samples),
        "avg_best_capacity": float(total_best_capacity_sum / total_samples),
        "avg_gap": float((total_best_capacity_sum - total_selected_capacity_sum) / total_samples),
        "best_selection_percentage": float(100.0 * total_selected_best_count / total_samples),
        "avg_num_switches_per_route": float(total_switches / total_routes),
        "total_switches": int(total_switches),
        "total_samples": int(total_samples),
        "n_routes": int(total_routes),
    }

    msg = (
        f"[epsilon={overall['epsilon']:.6f}] "
        f"inv_epsilon={overall['inv_epsilon']:.6f}, "
        f"avg_selected_capacity={overall['avg_selected_capacity']:.6f}, "
        f"avg_best_capacity={overall['avg_best_capacity']:.6f}, "
        f"avg_gap={overall['avg_gap']:.6f}, "
        f"best_selection_percentage={overall['best_selection_percentage']:.3f}%, "
        f"avg_num_switches_per_route={overall['avg_num_switches_per_route']:.3f}, "
        f"total_switches={overall['total_switches']}, "
        f"total_samples={overall['total_samples']}, "
        f"n_routes={overall['n_routes']}"
    )

    print(msg)
    if logger is not None:
        logger.info(msg)

    return overall


def sweep_epsilons(npz_files, epsilon_list, base_seed=42, logger=None):
    """
    Run a list of epsilons and return epsilon-level summaries.
    """
    results = []
    for eps in epsilon_list:
        overall = evaluate_epsilon_on_dataset(
            npz_files=npz_files,
            epsilon=eps,
            base_seed=base_seed,
            logger=logger,
        )
        results.append(overall)
    return results


###########################################################################
# Saving / plotting
###########################################################################
def save_epsilon_summary_npz(results, save_path):
    """
    Save only epsilon-level aggregated summary to NPZ.
    """
    if len(results) == 0:
        raise ValueError("results is empty, nothing to save.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    keys = results[0].keys()
    save_dict = {k: np.array([r[k] for r in results]) for k in keys}
    np.savez(save_path, **save_dict)


def save_epsilon_summary_json(results, save_path):
    """
    Optional readable backup.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def plot_inv_epsilon_vs_avg_capacity(results, save_path):
    x = [r["inv_epsilon"] for r in results]
    y = [r["avg_selected_capacity"] for r in results]

    plt.figure(figsize=(7, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("1 / epsilon")
    plt.ylabel("Average selected capacity")
    plt.title("1/epsilon vs average selected capacity")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


###########################################################################
# Logging
###########################################################################
def build_logger(log_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, mode="w")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)

    return logger


###########################################################################
# Main
###########################################################################
def main():
    # ---------------------------------------------------------------------
    # Config
    # ---------------------------------------------------------------------
    dataset_dir = Path("../../dataset/CQI_Mobility_medium_R_2_region_2")
    npz_files = sorted(dataset_dir.glob("*_result.npz"))
    n_routes = len(npz_files)

    if n_routes == 0:
        raise RuntimeError(f"No *_result.npz found in: {dataset_dir}")

    BASE_SEED = 42

    # 这里明确区分 integration_test
    experiment_tag = "epsilon_sweep_R_2"

    # 你后面也可以改成别的，比如：
    # experiment_tag = "ablation"
    # experiment_tag = "full_run"

    # epsilon_list = [1.0, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]
    epsilon_list = np.logspace(0, -2, 20)   # 从 1 → 0.01，20个点

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------------------------------------------------------------
    # Output paths
    # ---------------------------------------------------------------------
    out_root = os.path.join(
        BASE_DIR,
        f"{experiment_tag}_{timestamp}",
    )
    os.makedirs(out_root, exist_ok=True)

    log_path = os.path.join(out_root, f"results_epsilon_{timestamp}.log")
    npz_save_path = os.path.join(out_root, f"epsilon_sweep_summary_{timestamp}.npz")
    json_save_path = os.path.join(out_root, f"epsilon_sweep_summary_{timestamp}.json")
    fig_path = os.path.join(out_root, f"inv_epsilon_vs_avg_capacity_{timestamp}.png")

    logger = build_logger(log_path)

    print("############################################################")
    print(f"dataset dir: {dataset_dir}")
    print(f"Found {n_routes} result files")
    print(f"Experiment tag: {experiment_tag}")
    print(f"Output root: {out_root}")
    print(f"Log file: {log_path}")
    print(f"NPZ summary: {npz_save_path}")
    print(f"Plot file: {fig_path}")
    print("############################################################")

    logger.info("############################################################")
    logger.info(f"dataset_dir = {dataset_dir}")
    logger.info(f"n_routes = {n_routes}")
    logger.info(f"experiment_tag = {experiment_tag}")
    logger.info(f"output_root = {out_root}")
    logger.info(f"epsilon_list = {epsilon_list}")
    logger.info("############################################################")

    # ---------------------------------------------------------------------
    # Run sweep
    # ---------------------------------------------------------------------
    results = sweep_epsilons(
        npz_files=npz_files,
        epsilon_list=epsilon_list,
        base_seed=BASE_SEED,
        logger=logger,
    )

    # ---------------------------------------------------------------------
    # Save only epsilon-level aggregated info
    # ---------------------------------------------------------------------
    save_epsilon_summary_npz(results, npz_save_path)
    save_epsilon_summary_json(results, json_save_path)  # 可删，可保留
    plot_inv_epsilon_vs_avg_capacity(results, fig_path)

    logger.info(f"Saved epsilon-level summary npz: {npz_save_path}")
    logger.info(f"Saved epsilon-level summary json: {json_save_path}")
    logger.info(f"Saved figure: {fig_path}")

    print(f"Saved epsilon-level summary npz: {npz_save_path}")
    print(f"Saved epsilon-level summary json: {json_save_path}")
    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()