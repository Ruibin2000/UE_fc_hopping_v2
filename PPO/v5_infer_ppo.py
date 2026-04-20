from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional
from datetime import datetime
import itertools
import json
import math
import random
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# Config
# ============================================================

@dataclass
class InferenceConfig:
    # ------------------------------------------------------------------
    # Edit here
    # ------------------------------------------------------------------
    # run_dir: str = "./checkpoints_no_penalty/PPO_G1_lowCritic_risk_log1p_alpha0.1_eps0.3_lam10.0_lam2600.0_alr9e-05_clr5e-05_20260416_174102"
    run_dir: str = "./checkpoints_v5/PPO_G4_L1_verystrong_risk_log1p_alpha0.1_eps0.3_lam11.0_lam2600.0_alr9e-05_clr5e-05_20260417_033255"
    
    
    checkpoint_name: str = "best.pt"     # "best.pt" or "last.pt"

    split_name: str = "test"             # "train" / "val" / "test"
    max_routes: Optional[int] = None     # None => all routes in split

    deterministic: bool = False          # False = sample; True = greedy argmax
    save_per_route_npz: bool = True
    make_plots: bool = True

    # output folder name under run_dir
    inference_subdir: str = "inference_eval"

    # if you want to override device manually, set e.g. "cuda:0" or "cpu"
    device_override: Optional[str] = None

    # reproducibility
    seed: int = 42

    # ----------------------------------------------------------
    # Evaluation reward setting
    # reward_mode:
    #   None  => use checkpoint/train config reward_mode
    #   str   => override reward mode
    #
    # lambda rule:
    #   None  => use checkpoint/train lambda
    #   float => override lambda manually
    #
    # Note:
    #   This inference script only stores the checkpoint's own rollout
    #   results. Fair comparison under a fixed benchmark penalty should be
    #   done later in the plotting/metric script by recomputing reward from
    #   chosen_rate + z_trace.
    # ----------------------------------------------------------
    eval_reward_mode: Optional[str] = None

    eval_lambda1: Optional[float] = None
    eval_lambda2: Optional[float] = None
    
    eval_alpha_risk: Optional[float] = None
    eval_epsilon_budget: Optional[float] = None


cfg = InferenceConfig()


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_logger(output_dir: Path) -> logging.Logger:
    ensure_dir(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"infer_{timestamp}.log"

    logger = logging.getLogger("ppo_sep_inference")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, mode="w")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_path}")
    return logger


def build_action_list(n_arr: int) -> List[Tuple[int, ...]]:
    actions: List[Tuple[int, ...]] = []
    for i in range(n_arr):
        actions.append((i,))
    for i, j in itertools.combinations(range(n_arr), 2):
        actions.append((i, j))
    return actions


def build_action_sizes(action_list: Sequence[Tuple[int, ...]]) -> np.ndarray:
    return np.asarray([len(a) for a in action_list], dtype=np.int64)


def build_action_mask_from_z(
    z: np.ndarray,
    action_sizes: np.ndarray,
    threshold_b: float,
) -> np.ndarray:
    B = len(z)
    mask = np.ones((B, len(action_sizes)), dtype=np.float32)
    restricted = z >= threshold_b
    if restricted.any():
        single_only = (action_sizes == 1).astype(np.float32)
        mask[restricted] = single_only[None, :]
    return mask


def build_obs_input(
    obs_values: np.ndarray,
    obs_mask: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [obs_values, obs_mask, z[:, None].astype(np.float32)],
        axis=1,
    ).astype(np.float32)


def compute_reward(
    chosen: float,
    z_before: float,
    reward_mode: str,
    lambda1: float,
    lambda2: float,
) -> float:
    if reward_mode == "rate":
        return chosen

    if reward_mode == "risk_aware":
        penalty = lambda1 * (z_before - 1.0) * (
            chosen / (lambda2 + chosen + 1e-8)
        )
        return chosen * math.exp(-penalty)

    if reward_mode == "risk_log1p":
        penalty = lambda1 * (z_before - 1.0) * (
            chosen / (lambda2 + chosen + 1e-8)
        )
        return math.log1p(chosen) * math.exp(-penalty)

    raise ValueError(f"Unknown reward_mode: {reward_mode}")


# ============================================================
# Dataset
# ============================================================

def load_route_capacity(npz_path: Path) -> np.ndarray:
    """
    Expected raw shape after squeeze:
        [T, N_arr, N_cell]
    Internal shape:
        [T, N_cell, N_arr]
    """
    data = np.load(npz_path, allow_pickle=True)

    if "capacity" not in data:
        raise KeyError(f"'capacity' not found in {npz_path}")

    cap = np.squeeze(np.array(data["capacity"])).astype(float)

    if cap.ndim != 3:
        raise ValueError(f"Expected squeezed capacity to be 3D, got {cap.shape} in {npz_path}")

    cap = cap.transpose(0, 2, 1)
    return cap.astype(np.float32)


def load_all_routes(files: List[Path], expected_shape: Tuple[int, int, int]) -> List[np.ndarray]:
    routes: List[np.ndarray] = []
    for f in files:
        cap = load_route_capacity(f)
        if cap.shape != expected_shape:
            raise ValueError(
                f"Route shape mismatch for {f}\n"
                f"got {cap.shape}, expected {expected_shape}"
            )
        routes.append(cap)
    return routes


# ============================================================
# Model: separate actor / critic
# ============================================================

class SeparateRecurrentActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        actor_hidden_dim: int,
        actor_gru_hidden_dim: int,
        critic_hidden_dim: int,
        critic_gru_hidden_dim: int,
        num_actions: int,
    ):
        super().__init__()

        self.actor_encoder = nn.Sequential(
            nn.Linear(obs_dim, actor_hidden_dim),
            nn.ReLU(),
            nn.Linear(actor_hidden_dim, actor_hidden_dim),
            nn.ReLU(),
        )
        self.actor_gru = nn.GRU(
            input_size=actor_hidden_dim,
            hidden_size=actor_gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.policy_head = nn.Linear(actor_gru_hidden_dim, num_actions)

        self.critic_encoder = nn.Sequential(
            nn.Linear(obs_dim, critic_hidden_dim),
            nn.ReLU(),
            nn.Linear(critic_hidden_dim, critic_hidden_dim),
            nn.ReLU(),
        )
        self.critic_gru = nn.GRU(
            input_size=critic_hidden_dim,
            hidden_size=critic_gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.value_head = nn.Linear(critic_gru_hidden_dim, 1)

        self.actor_gru_hidden_dim = actor_gru_hidden_dim
        self.critic_gru_hidden_dim = critic_gru_hidden_dim

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h_actor = torch.zeros(1, batch_size, self.actor_gru_hidden_dim, device=device)
        h_critic = torch.zeros(1, batch_size, self.critic_gru_hidden_dim, device=device)
        return h_actor, h_critic

    def forward(
        self,
        obs_seq: torch.Tensor,
        h_actor0: torch.Tensor,
        h_critic0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a = self.actor_encoder(obs_seq)
        a_out, h_actorN = self.actor_gru(a, h_actor0)
        logits = self.policy_head(a_out)

        c = self.critic_encoder(obs_seq)
        c_out, h_criticN = self.critic_gru(c, h_critic0)
        values = self.value_head(c_out).squeeze(-1)

        return logits, values, h_actorN, h_criticN

    def forward_step(
        self,
        obs_t: torch.Tensor,
        h_actor: torch.Tensor,
        h_critic: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, h_actorN, h_criticN = self.forward(
            obs_t.unsqueeze(1), h_actor, h_critic
        )
        return logits[:, 0], values[:, 0], h_actorN, h_criticN


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def run_one_route(
    route: np.ndarray,                        # [T, N_cell, N_arr]
    model: SeparateRecurrentActorCritic,
    device: torch.device,
    action_list: Sequence[Tuple[int, ...]],
    action_sizes: np.ndarray,
    alpha_risk: float,
    epsilon_budget: float,
    reward_mode: str,
    lambda1: float,
    lambda2: float,
    deterministic: bool,
) -> Dict[str, np.ndarray | float]:
    T, n_cell, n_arr = route.shape
    threshold_b = 1.0 + epsilon_budget

    z = 1.0
    h_actor, h_critic = model.init_hidden(batch_size=1, device=device)

    prev_obs_values = np.zeros((1, n_cell * n_arr), dtype=np.float32)
    prev_obs_mask = np.zeros((1, n_cell * n_arr), dtype=np.float32)

    chosen_rates: List[float] = []
    global_best_rates: List[float] = []
    rewards: List[float] = []
    z_trace: List[float] = []
    action_indices: List[int] = []
    action_sizes_trace: List[int] = []
    restricted_flags: List[float] = []
    pair_flags: List[float] = []

    for t in range(T):
        obs_t = build_obs_input(
            prev_obs_values,
            prev_obs_mask,
            np.asarray([z], dtype=np.float32),
        )
        obs_tensor = torch.from_numpy(obs_t).to(device)

        logits, _, h_actor, h_critic = model.forward_step(obs_tensor, h_actor, h_critic)

        mask_np = build_action_mask_from_z(
            np.asarray([z], dtype=np.float32),
            action_sizes,
            threshold_b,
        )
        mask_t = torch.from_numpy(mask_np).to(device)
        masked_logits = logits.masked_fill(mask_t <= 0.0, -1e9)

        if deterministic:
            action_idx = int(torch.argmax(masked_logits, dim=-1).item())
        else:
            dist = Categorical(logits=masked_logits)
            action_idx = int(dist.sample().item())

        a_tuple = action_list[action_idx]
        rates_t = route[t]
        global_best = float(rates_t.max())

        obs_values = np.zeros((1, n_cell * n_arr), dtype=np.float32)
        obs_mask = np.zeros((1, n_cell * n_arr), dtype=np.float32)
        observed_vals = []

        for arr in a_tuple:
            for cell in range(n_cell):
                flat_idx = cell * n_arr + arr
                v = float(rates_t[cell, arr])
                obs_values[0, flat_idx] = v
                obs_mask[0, flat_idx] = 1.0
                observed_vals.append(v)

        chosen = float(max(observed_vals)) if observed_vals else 0.0

        reward = compute_reward(
            chosen=chosen,
            z_before=z,
            reward_mode=reward_mode,
            lambda1=lambda1,
            lambda2=lambda2,
        )

        chosen_rates.append(chosen)
        global_best_rates.append(global_best)
        rewards.append(reward)
        z_trace.append(z)
        action_indices.append(action_idx)
        action_sizes_trace.append(len(a_tuple))
        restricted_flags.append(1.0 if z >= threshold_b else 0.0)
        pair_flags.append(1.0 if len(a_tuple) == 2 else 0.0)

        z = (1.0 - alpha_risk) * z + alpha_risk * float(len(a_tuple))
        prev_obs_values = obs_values
        prev_obs_mask = obs_mask

    chosen_rates_np = np.asarray(chosen_rates, dtype=np.float32)
    global_best_rates_np = np.asarray(global_best_rates, dtype=np.float32)
    rewards_np = np.asarray(rewards, dtype=np.float32)
    z_trace_np = np.asarray(z_trace, dtype=np.float32)
    action_indices_np = np.asarray(action_indices, dtype=np.int64)
    action_sizes_np = np.asarray(action_sizes_trace, dtype=np.int64)
    restricted_np = np.asarray(restricted_flags, dtype=np.float32)
    pair_np = np.asarray(pair_flags, dtype=np.float32)

    route_summary = {
        "return_sum": float(rewards_np.sum()),
        "reward_mean": float(rewards_np.mean()),
        "chosen_rate_mean": float(chosen_rates_np.mean()),
        "best_rate_mean": float(global_best_rates_np.mean()),
        "best_ratio": float(chosen_rates_np.mean() / (global_best_rates_np.mean() + 1e-8)),
        "hit_rate": float(np.mean(np.isclose(chosen_rates_np, global_best_rates_np, atol=1e-6))),
        "pair_ratio": float(pair_np.mean()),
        "single_ratio": float(1.0 - pair_np.mean()),
        "z_mean": float(z_trace_np.mean()),
        "restricted_ratio": float(restricted_np.mean()),
    }

    return {
        "chosen_rate": chosen_rates_np,
        "global_best_rate": global_best_rates_np,
        "reward": rewards_np,
        "z_trace": z_trace_np,
        "action_idx": action_indices_np,
        "action_size": action_sizes_np,
        "restricted_flag": restricted_np,
        "summary": route_summary,
    }


def plot_z_curve(
    z_trace: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(10, 4))
    x = np.arange(len(z_trace))
    plt.plot(x, z_trace)
    plt.xlabel("Time step")
    plt.ylabel("z")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_chosen_and_z(
    chosen_rate: np.ndarray,
    global_best_rate: np.ndarray,
    z_trace: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(np.arange(len(chosen_rate)), chosen_rate, label="chosen_rate")
    ax1.plot(np.arange(len(global_best_rate)), global_best_rate, label="global_best_rate")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Rate")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)
    ax2.plot(np.arange(len(z_trace)), z_trace, label="z_trace")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("z")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    
def plot_full(
    route: np.ndarray,                 # [T, N_cell, N_arr]
    chosen_rate: np.ndarray,
    global_best_rate: np.ndarray,
    z_trace: np.ndarray,
    save_path: Path,
    title: str,
) -> None:
    T, n_cell, n_arr = route.shape
    x = np.arange(T)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1,
        figsize=(24, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 2]},
    )

    # =========================================================
    # 1️⃣ Top: raw（彩色，多条线）
    # =========================================================
    num_lines = n_cell * n_arr
    cmap = plt.cm.get_cmap("tab20", num_lines)  # 离散 colormap

    idx = 0
    for cell in range(n_cell):
        for arr in range(n_arr):
            ax1.plot(
                x,
                route[:, cell, arr],
                color=cmap(idx),
                linewidth=1.2,
            )
            idx += 1

    ax1.set_ylabel("Raw rate")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    # ❗ 不加 legend

    # =========================================================
    # 2️⃣ Middle: z（颜色 = chosen）
    # =========================================================
    chosen_line, = ax3.plot([], [])  # 先占位拿默认颜色
    chosen_color = chosen_line.get_color()
    chosen_line.remove()  # 删除占位

    ax2.plot(
        x,
        z_trace,
        label="z",
        linewidth=1.8,
        color=chosen_color,   # 👈 跟 chosen 一样
    )

    ax2.set_ylabel("z")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # =========================================================
    # 3️⃣ Bottom: chosen vs best
    # =========================================================
    ax3.plot(
        x,
        chosen_rate,
        label="chosen",
        linewidth=2.2,
        color=chosen_color,   # 👈 确保一致
    )

    ax3.plot(
        x,
        global_best_rate,
        linestyle="--",
        linewidth=1.8,
        alpha=0.8,
        label="best",
    )


    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Rate")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    
    
def save_route_npz(
    save_path: Path,
    result: Dict[str, np.ndarray | float],
    route_file: Path,
) -> None:
    summary = result["summary"]
    assert isinstance(summary, dict)

    np.savez_compressed(
        save_path,
        route_file=str(route_file),
        chosen_rate=result["chosen_rate"],
        global_best_rate=result["global_best_rate"],
        reward=result["reward"],
        z_trace=result["z_trace"],
        action_idx=result["action_idx"],
        action_size=result["action_size"],
        restricted_flag=result["restricted_flag"],
        chosen_rate_mean=np.float32(summary["chosen_rate_mean"]),
        best_rate_mean=np.float32(summary["best_rate_mean"]),
        best_ratio=np.float32(summary["best_ratio"]),
        hit_rate=np.float32(summary["hit_rate"]),
        pair_ratio=np.float32(summary["pair_ratio"]),
        single_ratio=np.float32(summary["single_ratio"]),
        z_mean=np.float32(summary["z_mean"]),
        restricted_ratio=np.float32(summary["restricted_ratio"]),
        return_sum=np.float32(summary["return_sum"]),
        reward_mean=np.float32(summary["reward_mean"]),
    )


def aggregate_route_summaries(route_summaries: List[Dict[str, float]]) -> Dict[str, float]:
    if not route_summaries:
        raise ValueError("No route summaries to aggregate.")

    return {
        "num_routes": len(route_summaries),
        "mean_return": float(np.mean([x["return_sum"] for x in route_summaries])),
        "mean_reward_per_step": float(np.mean([x["reward_mean"] for x in route_summaries])),
        "mean_chosen_rate": float(np.mean([x["chosen_rate_mean"] for x in route_summaries])),
        "mean_best_rate": float(np.mean([x["best_rate_mean"] for x in route_summaries])),
        "mean_hit_global_best_rate": float(np.mean([x["hit_rate"] for x in route_summaries])),
        "best_ratio": float(np.mean([x["best_ratio"] for x in route_summaries])),
        "pair_ratio": float(np.mean([x["pair_ratio"] for x in route_summaries])),
        "single_ratio": float(np.mean([x["single_ratio"] for x in route_summaries])),
        "z_mean": float(np.mean([x["z_mean"] for x in route_summaries])),
        "restricted_ratio": float(np.mean([x["restricted_ratio"] for x in route_summaries])),
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    set_seed(cfg.seed)

    run_dir = Path(cfg.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found: {config_path}")

    checkpoint_path = run_dir / cfg.checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    output_name = f"{cfg.split_name}_{cfg.checkpoint_name.replace('.pt', '')}"
    output_dir = run_dir / cfg.inference_subdir / output_name
    per_route_npz_dir = output_dir / "route_npz"
    per_route_plot_dir = output_dir / "route_plots"

    ensure_dir(output_dir)
    ensure_dir(per_route_npz_dir)
    ensure_dir(per_route_plot_dir)

    logger = setup_logger(output_dir)

    with open(config_path, "r", encoding="utf-8") as f:
        train_cfg = json.load(f)

    if "split_files" not in train_cfg:
        raise KeyError("split_files not found in config.json")

    if cfg.split_name not in train_cfg["split_files"]:
        raise KeyError(f"Unknown split_name: {cfg.split_name}")

    split_files = [Path(p) for p in train_cfg["split_files"][cfg.split_name]]
    if cfg.max_routes is not None:
        split_files = split_files[:cfg.max_routes]

    if not split_files:
        raise RuntimeError(f"No files found for split={cfg.split_name}")

    sample = load_route_capacity(split_files[0])
    T, n_cell, n_arr = sample.shape
    routes = load_all_routes(split_files, (T, n_cell, n_arr))

    action_list = build_action_list(n_arr)
    action_sizes = build_action_sizes(action_list)
    num_actions = len(action_list)

    obs_dim = 2 * n_cell * n_arr + 1

    device_str = cfg.device_override if cfg.device_override is not None else train_cfg["device"]
    device = torch.device(device_str)

    model = SeparateRecurrentActorCritic(
        obs_dim=obs_dim,
        actor_hidden_dim=int(train_cfg["actor_hidden_dim"]),
        actor_gru_hidden_dim=int(train_cfg["actor_gru_hidden_dim"]),
        critic_hidden_dim=int(train_cfg["critic_hidden_dim"]),
        critic_gru_hidden_dim=int(train_cfg["critic_gru_hidden_dim"]),
        num_actions=num_actions,
    ).to(device)

    payload = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    eval_reward_mode = (
        cfg.eval_reward_mode
        if cfg.eval_reward_mode is not None
        else str(train_cfg["reward_mode"])
    )

    train_lambda1 = float(train_cfg["lambda1"])
    train_lambda2 = float(train_cfg["lambda2"])

    if cfg.eval_lambda1 is None:
        eval_lambda1 = train_lambda1
    else:
        eval_lambda1 = float(cfg.eval_lambda1)

    if cfg.eval_lambda2 is None:
        eval_lambda2 = train_lambda2
    else:
        eval_lambda2 = float(cfg.eval_lambda2)

    if cfg.eval_lambda1 is None and cfg.eval_lambda2 is None:
        lambda_source = "checkpoint"
    else:
        lambda_source = "override"



    eval_alpha_risk = (
        cfg.eval_alpha_risk
        if cfg.eval_alpha_risk is not None
        else float(train_cfg["alpha_risk"])
    )
    eval_epsilon_budget = (
        cfg.eval_epsilon_budget
        if cfg.eval_epsilon_budget is not None
        else float(train_cfg["epsilon_budget"])
    )

    logger.info(f"Run dir           : {run_dir}")
    logger.info(f"Checkpoint        : {checkpoint_path.name}")
    logger.info(f"Split             : {cfg.split_name}")
    logger.info(f"Num routes        : {len(routes)}")
    logger.info(f"Deterministic     : {cfg.deterministic}")
    logger.info(f"Device            : {device}")
    logger.info(f"Output dir        : {output_dir}")
    logger.info(f"Dataset shape     : T={T}, N_cell={n_cell}, N_arr={n_arr}")
    logger.info(f"Eval reward_mode  : {eval_reward_mode}")
    logger.info(f"Train lambda1     : {train_lambda1}")
    logger.info(f"Train lambda2     : {train_lambda2}")
    logger.info(f"Eval lambda1      : {eval_lambda1}")
    logger.info(f"Eval lambda2      : {eval_lambda2}")
    logger.info(f"Lambda source     : {lambda_source}")
    logger.info(f"Eval alpha_risk   : {eval_alpha_risk}")
    logger.info(f"Eval eps_budget   : {eval_epsilon_budget}")
    logger.info(f"Seed              : {cfg.seed}")

    route_summaries: List[Dict[str, float]] = []

    for route_idx, (route_file, route) in enumerate(tqdm(list(zip(split_files, routes)), desc="Inference")):
        result = run_one_route(
            route=route,
            model=model,
            device=device,
            action_list=action_list,
            action_sizes=action_sizes,
            alpha_risk=eval_alpha_risk,
            epsilon_budget=eval_epsilon_budget,
            reward_mode=eval_reward_mode,
            lambda1=eval_lambda1,
            lambda2=eval_lambda2,
            deterministic=cfg.deterministic,
        )

        summary = result["summary"]
        assert isinstance(summary, dict)
        route_summaries.append(summary)

        stem = route_file.stem
        save_path = per_route_npz_dir / f"ppo_{stem}.npz"

        if cfg.save_per_route_npz:
            save_route_npz(
                save_path=save_path,
                result=result,
                route_file=route_file,
            )

        if cfg.make_plots:
            z_trace = result["z_trace"]
            chosen_rate = result["chosen_rate"]
            global_best_rate = result["global_best_rate"]

            assert isinstance(z_trace, np.ndarray)
            assert isinstance(chosen_rate, np.ndarray)
            assert isinstance(global_best_rate, np.ndarray)

            # plot_z_curve(
            #     z_trace=z_trace,
            #     save_path=per_route_plot_dir / f"{stem}_z.png",
            #     title=f"{stem}: z trace",
            # )

            # plot_chosen_and_z(
            #     chosen_rate=chosen_rate,
            #     global_best_rate=global_best_rate,
            #     z_trace=z_trace,
            #     save_path=per_route_plot_dir / f"{stem}_chosen_and_z.png",
            #     title=f"{stem}: chosen rate / best rate / z",
            # )
            
            plot_full(
                route=route,
                chosen_rate=chosen_rate,
                global_best_rate=global_best_rate,
                z_trace=z_trace,
                save_path=per_route_plot_dir / f"{stem}_full3.png",
                title=f"{stem}",
            )
            

        logger.info(
            f"[Route {route_idx + 1}/{len(routes)}] "
            f"{route_file.name} | "
            f"reward_mean={summary['reward_mean']:.4f} | "
            f"return_sum={summary['return_sum']:.4f} | "
            f"chosen_rate_mean={summary['chosen_rate_mean']:.4f} | "
            f"best_ratio={summary['best_ratio']:.4f} | "
            f"hit_rate={summary['hit_rate']:.4f} | "
            f"pair_ratio={summary['pair_ratio']:.4f} | "
            f"z_mean={summary['z_mean']:.4f}"
        )

    final_stats = aggregate_route_summaries(route_summaries)

    logger.info("=== INFERENCE DONE ===")
    logger.info(json.dumps(final_stats, indent=2))

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(final_stats, f, indent=2)

    np.savez_compressed(
        output_dir / "all_routes_summary.npz",
        num_routes=np.int32(final_stats["num_routes"]),
        mean_return=np.float32(final_stats["mean_return"]),
        mean_reward_per_step=np.float32(final_stats["mean_reward_per_step"]),
        mean_chosen_rate=np.float32(final_stats["mean_chosen_rate"]),
        mean_best_rate=np.float32(final_stats["mean_best_rate"]),
        mean_hit_global_best_rate=np.float32(final_stats["mean_hit_global_best_rate"]),
        best_ratio=np.float32(final_stats["best_ratio"]),
        pair_ratio=np.float32(final_stats["pair_ratio"]),
        single_ratio=np.float32(final_stats["single_ratio"]),
        z_mean=np.float32(final_stats["z_mean"]),
        restricted_ratio=np.float32(final_stats["restricted_ratio"]),
    )


if __name__ == "__main__":
    main()