from pathlib import Path
from datetime import datetime
import os
import logging

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm


###########################################################################
# Core algorithm: link-level semi-bandit over (RX, TX)
###########################################################################
def bandit_link_single_ue(capacity, epsilon=0.2, alpha=None, seed=None, verbose=False):
    """
    Link-level semi-bandit for a single UE.

    Action space
    ------------
    arm = (rx, tx), so total number of arms = n_rx * n_tx

    Rules
    -----
    t = 0:
        Full observation of all links.
        Update all observed links.
        Select the best observed link as the initial serving link.

    t >= 1, EXPLORE:
        - Keep current serving RX open, observe all TX links under this RX
        - Randomly choose another RX (different from current RX),
          observe all TX links under that RX
        - Update all observed links
        - Select the best link among the observed links as serving link

    t >= 1, EXPLOIT:
        - Select the best link according to Q over all arms
        - Observe and update only that chosen arm

    Parameters
    ----------
    capacity : np.ndarray
        Shape (N_sample, N_RX, N_TX)
    epsilon : float
        Exploration probability.
    alpha : float or None
        If None, use sample-average update.
        If float in (0, 1], use constant-step update.
    seed : int or None
        Random seed.
    verbose : bool
        If True, print per-step debug info.

    Returns
    -------
    dict
        {
            "serving_rx":       (N_sample,),
            "serving_tx":       (N_sample,),
            "serving_capacity": (N_sample,),
            "explore_flag":     (N_sample,),
            "switch_flag":      (N_sample,),
        }
    """
    capacity = np.asarray(capacity)
    if capacity.ndim != 3:
        raise ValueError("capacity must have shape (N_sample, N_RX, N_TX)")

    if not (0.0 <= epsilon <= 1.0):
        raise ValueError("epsilon must be in [0, 1]")

    if alpha is not None and not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1] when provided")

    n_sample, n_rx, n_tx = capacity.shape
    rng = np.random.default_rng(seed)

    # Q and N for each link arm = (rx, tx)
    Q = np.zeros((n_rx, n_tx), dtype=np.float64)
    N = np.zeros((n_rx, n_tx), dtype=np.int64)

    serving_rx = np.full(n_sample, -1, dtype=int)
    serving_tx = np.full(n_sample, -1, dtype=int)
    serving_capacity = np.zeros(n_sample, dtype=capacity.dtype)

    explore_flag = np.zeros(n_sample, dtype=bool)
    switch_flag = np.zeros(n_sample, dtype=bool)

    def update_q(rx, tx, reward):
        """Update one link arm (rx, tx)."""
        N[rx, tx] += 1
        if alpha is None:
            Q[rx, tx] += (float(reward) - Q[rx, tx]) / N[rx, tx]
        else:
            Q[rx, tx] = (1.0 - alpha) * Q[rx, tx] + alpha * float(reward)

    # ---------------------------------------------------------------------
    # t = 0: full observation initialization
    # ---------------------------------------------------------------------
    mat0 = capacity[0]  # shape (n_rx, n_tx)

    # Update all observed links
    for rx in range(n_rx):
        for tx in range(n_tx):
            update_q(rx, tx, mat0[rx, tx])

    # Select best observed link as initial serving link
    flat0 = int(np.argmax(mat0))
    rx0, tx0 = np.unravel_index(flat0, mat0.shape)
    reward0 = mat0[rx0, tx0]

    serving_rx[0] = rx0
    serving_tx[0] = tx0
    serving_capacity[0] = reward0

    if verbose:
        print("[t=0] INIT (full observation)")
        print(f"  observed links:\n{mat0}")
        print(f"  selected: RX={rx0}, TX={tx0}, cap={reward0:.4f}")
        print(f"  Q=\n{Q}")
        print(f"  N=\n{N}")

    # ---------------------------------------------------------------------
    # t >= 1
    # ---------------------------------------------------------------------
    for t in range(1, n_sample):
        prev_rx = serving_rx[t - 1]
        prev_tx = serving_tx[t - 1]

        if rng.random() < epsilon:
            # =============================================================
            # EXPLORE:
            #   - keep current serving RX open
            #   - randomly choose another RX
            #   - observe all TX links on both RXs
            #   - update all observed links
            #   - choose the best link among the observed links
            # =============================================================
            explore_flag[t] = True

            current_rx = prev_rx
            candidate_rxs = [r for r in range(n_rx) if r != current_rx]
            probe_rx = int(rng.choice(candidate_rxs))

            current_links = capacity[t, current_rx, :]   # shape (n_tx,)
            probe_links = capacity[t, probe_rx, :]       # shape (n_tx,)

            # Update all observed links
            for tx in range(n_tx):
                update_q(current_rx, tx, current_links[tx])
                update_q(probe_rx, tx, probe_links[tx])

            # Choose best among observed links
            current_best_tx = int(np.argmax(current_links))
            probe_best_tx = int(np.argmax(probe_links))

            current_best_reward = current_links[current_best_tx]
            probe_best_reward = probe_links[probe_best_tx]

            if current_best_reward >= probe_best_reward:
                rx = current_rx
                tx = current_best_tx
                reward = current_best_reward
            else:
                rx = probe_rx
                tx = probe_best_tx
                reward = probe_best_reward

            if verbose:
                print(f"\n[t={t}] EXPLORE")
                print(f"  current_rx={current_rx}, links={current_links}")
                print(f"  probe_rx={probe_rx}, links={probe_links}")
                print("  observed arms:")
                for j in range(n_tx):
                    print(f"    ({current_rx},{j}) -> {current_links[j]:.4f}")
                for j in range(n_tx):
                    print(f"    ({probe_rx},{j}) -> {probe_links[j]:.4f}")
                print(f"  selected from observed links: RX={rx}, TX={tx}, cap={reward:.4f}")

        else:
            # =============================================================
            # EXPLOIT:
            #   - choose best arm according to Q over all links
            #   - observe/update only the chosen arm
            # =============================================================
            flat_idx = int(np.argmax(Q))
            rx, tx = np.unravel_index(flat_idx, Q.shape)

            reward = capacity[t, rx, tx]
            update_q(rx, tx, reward)

            if verbose:
                print(f"\n[t={t}] EXPLOIT")
                print(f"  selected arm from Q: RX={rx}, TX={tx}, cap={reward:.4f}")

        serving_rx[t] = rx
        serving_tx[t] = tx
        serving_capacity[t] = reward
        switch_flag[t] = (rx != prev_rx) or (tx != prev_tx)

        if verbose:
            print(f"  switched={switch_flag[t]}")
            print(f"  Q=\n{Q}")
            print(f"  N=\n{N}")

    return {
        "serving_rx": serving_rx,
        "serving_tx": serving_tx,
        "serving_capacity": serving_capacity,
        "explore_flag": explore_flag,
        "switch_flag": switch_flag,
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
        Output of bandit_link_single_ue(...)

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

    best_capacity = np.max(capacity, axis=(1, 2))

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

    if cap.dtype == object:
        cap = np.stack(cap, axis=0)
    cap = np.squeeze(cap, axis=(3, 5, 6))

    if sinr_db.dtype == object:
        sinr_db = np.stack(sinr_db, axis=0)
    sinr_db = np.squeeze(sinr_db, axis=(3, 5, 6))

    return cap, sinr_db


###########################################################################
# Helpers
###########################################################################
def alpha_to_str(alpha):
    """Human-readable alpha tag."""
    return "None(sample_avg)" if alpha is None else f"{alpha:.6f}"


def alpha_to_save_value(alpha):
    """Numeric value for npz save."""
    return np.nan if alpha is None else float(alpha)


###########################################################################
# Dataset-level aggregation
###########################################################################
def evaluate_bandit_on_dataset(npz_files, epsilon, alpha=None, base_seed=42, logger=None):
    """
    Run one alpha over the whole dataset and only keep alpha-level summary.

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

    alpha_str = alpha_to_str(alpha)

    for route_idx, npz_path in enumerate(
        tqdm(npz_files, desc=f"bandit alpha={alpha_str}", ncols=100)
    ):
        capacity, sinr_db = load_channel(npz_path)

        # keep only one UE dimension
        capacity = capacity[:, 0, :]   # -> (N_sample, N_RX, N_TX)
        _ = sinr_db[:, 0, :]

        res = bandit_link_single_ue(
            capacity=capacity,
            epsilon=epsilon,
            alpha=alpha,
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
        raise RuntimeError("No samples found when evaluating bandit.")

    overall = {
        "epsilon": float(epsilon),
        "inv_epsilon": float(1.0 / epsilon) if epsilon > 0 else np.inf,
        "alpha": alpha_to_save_value(alpha),
        "alpha_is_none": bool(alpha is None),
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
        f"[bandit alpha={alpha_str}] "
        f"epsilon={overall['epsilon']:.6f}, "
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


def sweep_bandit_alphas(npz_files, epsilon, alpha_list, base_seed=42, logger=None):
    """
    Run a list of alphas and return alpha-level summaries.
    alpha_list can contain None.
    """
    results = []
    for alpha in alpha_list:
        overall = evaluate_bandit_on_dataset(
            npz_files=npz_files,
            epsilon=epsilon,
            alpha=alpha,
            base_seed=base_seed,
            logger=logger,
        )
        results.append(overall)
    return results


###########################################################################
# Saving / plotting
###########################################################################
def save_bandit_summary_npz(results, save_path):
    """
    Save alpha-level aggregated summary to NPZ.
    alpha=None is stored as:
      - alpha = np.nan
      - alpha_is_none = True
    """
    if len(results) == 0:
        raise ValueError("results is empty, nothing to save.")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    keys = results[0].keys()
    save_dict = {}
    for k in keys:
        vals = [r[k] for r in results]
        if isinstance(vals[0], (bool, np.bool_)):
            save_dict[k] = np.array(vals, dtype=bool)
        else:
            save_dict[k] = np.array(vals)

    np.savez(save_path, **save_dict)


def plot_alpha_vs_avg_capacity(results, save_path):
    """
    Plot alpha vs avg_selected_capacity.
    alpha=None is placed at x=0 and labeled 'None'.
    """
    x = []
    y = []
    labels = []

    for r in results:
        if r["alpha_is_none"]:
            x.append(0.0)
            labels.append("None")
        else:
            x.append(float(r["alpha"]))
            labels.append(f"{float(r['alpha']):g}")
        y.append(r["avg_selected_capacity"])

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("alpha (None shown as 0)")
    plt.ylabel("Average selected capacity")
    plt.title("Bandit Link: alpha vs average selected capacity")
    plt.grid(True, alpha=0.3)

    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_alpha_vs_best_selection(results, save_path):
    """
    Plot alpha vs best_selection_percentage.
    alpha=None is placed at x=0 and labeled 'None'.
    """
    x = []
    y = []
    labels = []

    for r in results:
        if r["alpha_is_none"]:
            x.append(0.0)
            labels.append("None")
        else:
            x.append(float(r["alpha"]))
            labels.append(f"{float(r['alpha']):g}")
        y.append(r["best_selection_percentage"])

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel("alpha (None shown as 0)")
    plt.ylabel("Best selection percentage (%)")
    plt.title("Bandit Link: alpha vs best selection percentage")
    plt.grid(True, alpha=0.3)

    for xi, yi, lab in zip(x, y, labels):
        plt.annotate(lab, (xi, yi), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=8)

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
    BASE_SEED = 42
    epsilon = 0.2

    dataset_dir = Path("../../dataset/CQI_Mobility_medium_R_2_region_2")
    npz_files = sorted(dataset_dir.glob("*_result.npz"))
    n_routes = len(npz_files)

    if n_routes == 0:
        raise RuntimeError(f"No *_result.npz found in: {dataset_dir}")

    # alpha sweep, including None
    alpha_list = [None, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

    experiment_tag = f"bandit_link_alpha_sweep_R_2_eps_{epsilon}"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------------------------------------------------------------
    # Output paths
    # ---------------------------------------------------------------------
    out_root = os.path.join(BASE_DIR, f"{experiment_tag}_{timestamp}")
    os.makedirs(out_root, exist_ok=True)

    log_path = os.path.join(out_root, f"results_bandit_link_alpha_{timestamp}.log")
    npz_save_path = os.path.join(out_root, f"bandit_link_alpha_sweep_summary_{timestamp}.npz")
    fig_alpha_cap_path = os.path.join(out_root, f"alpha_vs_avg_capacity_{timestamp}.png")
    fig_alpha_best_path = os.path.join(out_root, f"alpha_vs_best_selection_{timestamp}.png")

    logger = build_logger(log_path)

    print("############################################################")
    print(f"dataset dir: {dataset_dir}")
    print(f"Found {n_routes} result files")
    print(f"Experiment tag: {experiment_tag}")
    print(f"Output root: {out_root}")
    print(f"Log file: {log_path}")
    print(f"NPZ summary: {npz_save_path}")
    print(f"Figure (alpha vs avg capacity): {fig_alpha_cap_path}")
    print(f"Figure (alpha vs best selection): {fig_alpha_best_path}")
    print(f"fixed epsilon: {epsilon}")
    print(f"alpha_list: {alpha_list}")
    print("############################################################")

    logger.info("############################################################")
    logger.info(f"dataset_dir = {dataset_dir}")
    logger.info(f"n_routes = {n_routes}")
    logger.info(f"experiment_tag = {experiment_tag}")
    logger.info(f"output_root = {out_root}")
    logger.info(f"fixed epsilon = {epsilon}")
    logger.info(f"alpha_list = {alpha_list}")
    logger.info("############################################################")

    # ---------------------------------------------------------------------
    # Run sweep
    # ---------------------------------------------------------------------
    results = sweep_bandit_alphas(
        npz_files=npz_files,
        epsilon=epsilon,
        alpha_list=alpha_list,
        base_seed=BASE_SEED,
        logger=logger,
    )

    # ---------------------------------------------------------------------
    # Save / plot
    # ---------------------------------------------------------------------
    save_bandit_summary_npz(results, npz_save_path)
    plot_alpha_vs_avg_capacity(results, fig_alpha_cap_path)
    plot_alpha_vs_best_selection(results, fig_alpha_best_path)

    logger.info(f"Saved bandit summary npz: {npz_save_path}")
    logger.info(f"Saved figure: {fig_alpha_cap_path}")
    logger.info(f"Saved figure: {fig_alpha_best_path}")

    print(f"Saved bandit summary npz: {npz_save_path}")
    print(f"Saved figure: {fig_alpha_cap_path}")
    print(f"Saved figure: {fig_alpha_best_path}")


if __name__ == "__main__":
    main()