from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import logging
from tqdm import tqdm
from datetime import datetime


###########################################################################
# helper functions
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
    results : dict
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

    N_sample, N_RX, N_TX = capacity.shape
    rng = np.random.default_rng(seed)

    serving_rx = np.full(N_sample, -1, dtype=int)
    serving_tx = np.full(N_sample, -1, dtype=int)
    serving_capacity = np.zeros(N_sample, dtype=capacity.dtype)

    explore_flag = np.zeros(N_sample, dtype=bool)
    switch_flag = np.zeros(N_sample, dtype=bool)
    new_rx_record = np.full(N_sample, np.nan, dtype=float)

    # --------------------------------------------------
    # t = 0: open all RX, choose global best (RX, TX)
    # --------------------------------------------------
    mat0 = capacity[0]  # (N_RX, N_TX)
    flat_idx = np.argmax(mat0)
    rx0, tx0 = np.unravel_index(flat_idx, mat0.shape)

    serving_rx[0] = rx0
    serving_tx[0] = tx0
    serving_capacity[0] = mat0[rx0, tx0]

    if verbose:
        print(f"[t=0] INIT")
        print(f"  selected: RX={rx0}, TX={tx0}, cap={serving_capacity[0]:.4f}")

    # --------------------------------------------------
    # t >= 1
    # --------------------------------------------------
    for t in range(1, N_sample):
        prev_rx = serving_rx[t - 1]
        prev_tx = serving_tx[t - 1]

        # Keep previous serving link
        if rng.random() <= (1.0 - epsilon):
            serving_rx[t] = prev_rx
            serving_tx[t] = prev_tx
            serving_capacity[t] = capacity[t, prev_rx, prev_tx]

        # Explore: open one new RX, compare old RX and new RX
        else:
            explore_flag[t] = True

            candidate_rxs = [rx for rx in range(N_RX) if rx != prev_rx]
            rx_new = rng.choice(candidate_rxs)
            new_rx_record[t] = float(rx_new)

            rx_pair = [prev_rx, rx_new]
            submat = capacity[t, rx_pair, :]  # shape (2, N_TX)

            # Compare both old RX and new RX over all TX
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
                print(f"  prev serving: RX={prev_rx}, TX={prev_tx}, "
                      f"cap_now={capacity[t, prev_rx, prev_tx]:.4f}")
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


def plot_selected_vs_best(
    capacity,
    serving_capacity,
    switch_flag=None,
    figsize=(15, 5),
    save_dir="plots",
    filename="selected_vs_best.png",
    dpi=200,
    show=False,
):
    """
    Save plot to a folder (auto-create if not exists).
    """

    # ✅ 自动创建文件夹
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)

    capacity = np.asarray(capacity)
    serving_capacity = np.asarray(serving_capacity)

    N_sample = capacity.shape[0]
    t = np.arange(N_sample)

    global_best = np.max(capacity, axis=(1, 2))

    plt.figure(figsize=figsize)
    plt.plot(t, global_best, linewidth=1.5, label="Global best capacity")
    plt.plot(t, serving_capacity, linewidth=1.5, label="Selected serving capacity")

    if switch_flag is not None:
        switch_flag = np.asarray(switch_flag).astype(bool)
        idx = np.where(switch_flag)[0]
        if len(idx) > 0:
            plt.scatter(
                idx,
                serving_capacity[idx],
                s=35,
                marker="o",
                c="black",
                label="RX switch",
            )

    plt.xlabel("Time step")
    plt.ylabel("Capacity")
    plt.title("Selected capacity vs global best capacity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()

    gap = global_best - serving_capacity
    # print(f"Saved figure to: {save_path}")
    # print(f"Average gap to global best: {np.mean(gap):.4f}")
    # print(f"Max gap to global best: {np.max(gap):.4f}")

def summarize_selection_performance(capacity, res, npz_path, verbose=False, logger=None):
    """
    capacity: (N_sample, N_RX, N_TX)
    res: output dict from sticky_rx_single_ue()
    """

    capacity = np.asarray(capacity)

    serving_rx = np.asarray(res["serving_rx"])
    serving_tx = np.asarray(res["serving_tx"])
    serving_capacity = np.asarray(res["serving_capacity"])
    switch_flag = np.asarray(res["switch_flag"]).astype(bool)

    N_sample, N_RX, N_TX = capacity.shape

    # global best capacity
    best_capacity = np.max(capacity, axis=(1, 2))

    # global best link
    best_flat_idx = np.argmax(capacity.reshape(N_sample, -1), axis=1)
    best_rx, best_tx = np.unravel_index(best_flat_idx, (N_RX, N_TX))

    # whether selected the global best link
    selected_best_flag = (serving_rx == best_rx) & (serving_tx == best_tx)

    # statistics
    best_selection_percentage = 100.0 * np.mean(selected_best_flag)
    num_switches = int(np.sum(switch_flag))
    avg_best_capacity = float(np.mean(best_capacity))
    avg_selected_capacity = float(np.mean(serving_capacity))
    avg_gap = float(np.mean(best_capacity - serving_capacity))
    max_gap = float(np.max(best_capacity - serving_capacity))

    stats = {
        "best_selection_percentage": best_selection_percentage,
        "num_switches": num_switches,
        "avg_best_capacity": avg_best_capacity,
        "avg_selected_capacity": avg_selected_capacity,
        "avg_gap": avg_gap,
        "max_gap": max_gap,
        "selected_best_flag": selected_best_flag,
        "best_capacity": best_capacity,
        "best_rx": best_rx,
        "best_tx": best_tx,
    }

    # -------- 输出字符串 --------
    msg = (
        f"=== {npz_path.stem} ===\n"
        f"Selected global best link: {best_selection_percentage:.2f}%\n"
        f"Number of RX switches: {num_switches}\n"
        f"Average global best capacity: {avg_best_capacity:.4f}\n"
        f"Average selected capacity: {avg_selected_capacity:.4f}\n"
        f"Average gap: {avg_gap:.4f}\n"
        f"Max gap: {max_gap:.4f}"
    )

    # -------- logging --------
    if logger is not None:
        logger.info("\n" + msg)

    # # -------- 控制台打印 --------
    # if verbose:
    #     print(msg)

    return stats

def load_channel(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    cap = data["capacity"]
    sinr_db = data["sinr_db"]

    # object array -> stack
    if cap.dtype == object:
        cap = np.stack(cap, axis=0)
    cap = np.squeeze(cap, axis=(3,5,6))
    
    if sinr_db.dtype == object:
        sinr_db = np.stack(sinr_db, axis=0)
    sinr_db = np.squeeze(sinr_db, axis=(3,5,6))
    return cap, sinr_db

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import logging
from tqdm import tqdm

def main():

    dataset_dir = Path("../dataset/CQI_Mobility_medium_region_2")   # 你存 result.npz 的目录
    npz_files = sorted(dataset_dir.glob("*_result.npz"))
    # verbose = True
    verbose = False
    n_routes = len(npz_files)

    BASE_SEED = 42

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    plot_dir = os.path.join(BASE_DIR, "plots", "epsilon_greedy_single_ue")
    os.makedirs(plot_dir, exist_ok=True)
    
    selection_res_dir = os.path.join(BASE_DIR, "selection_results", "epsilon_greedy_single_ue")
    os.makedirs(selection_res_dir, exist_ok=True)


    # log 文件路径
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = os.path.join(BASE_DIR, f"results_epsilon_{timestamp}.log")

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # ❗关键两行（避免重复 + 不继承 root logger）
    logger.handlers.clear()
    logger.propagate = False

    # ✅ 覆盖写（每次运行重置 log）
    file_handler = logging.FileHandler(log_path, mode="w")

    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )

    # 如果你想要终端输出（可选）
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )

    # 添加 handler
    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)   # ❗不想终端输出就删掉这一行

    print("############################################################")
    print(f"dataset dir: {dataset_dir}")
    print(f"Found {n_routes} result files")
    print(f"Debug: {verbose}")
    print(f"Plots will be saved to: {plot_dir}")
    print(f"Selection results will be saved to: {selection_res_dir}")
    print(f"Log file: {log_path}")
    print("############################################################")


    for npz_path in tqdm(npz_files, desc="Routes", ncols=100):
        # logger.info(f"=== {npz_path.stem} ===")
        
        capacity, sinr_db = load_channel(npz_path)
        capacity = capacity[:, 0, :]
        sinr_db = sinr_db[:, 0, :]

        if verbose:
            print("------------------------------------------------")
            print(f"Processing file: {npz_path.name}")
            print("capacity shape:", capacity.shape)
            print("sinr_db shape:", sinr_db.shape)
            print(f"Number of time steps: {capacity.shape[0]}")
            print("------------------------------------------------")
        
        epsilon_res = epsilon_single_ue(
            capacity=capacity,
            epsilon=0.3,
            seed=BASE_SEED,
            verbose=False
        )
        
        plot_selected_vs_best(
            capacity,
            epsilon_res["serving_capacity"],
            switch_flag=epsilon_res["switch_flag"],
            save_dir=plot_dir,
            filename=f"{npz_path.stem}.png",
        )
        
        stats = summarize_selection_performance(
            capacity,
            epsilon_res,
            npz_path,
            verbose=False,
            logger=logger,
        )
        
        
        epsilon_res_path = os.path.join(selection_res_dir, f"{npz_path.stem}_res.npz")

        np.savez(
            epsilon_res_path,
            **epsilon_res
        )
        logger.info(f"Saved selection result: {epsilon_res_path}")
        
if __name__ == "__main__":
    main()