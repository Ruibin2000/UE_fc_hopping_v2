from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Sequence, Optional
from datetime import datetime
import itertools
import json
import math
import random
import time
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


# ============================================================
# Config
# ============================================================

@dataclass
class Config:
    seed: int = 42

    # dataset
    dataset_dir: str = "../dataset/CQI_arr8_medium_R_2_region_3"

    # output
    output_dir: str = "./checkpoints_v4"

    # split
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # train sampling
    num_envs: int = 16
    burn_in: int = 64
    train_horizon: int = 256

    # PPO
    total_updates: int = 2000
    update_epochs: int = 4
    minibatch_size: int = 8
    clip_eps: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 1e-4
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.01
    value_coef: float = 0.1

    # auxiliary observation prediction loss
    obs_loss_coef: float = 0.02

    # model
    encoder_hidden_dim: int = 128
    gru_hidden_dim: int = 128

    # risk dynamics
    alpha_risk: float = 0.1
    epsilon_budget: float = 0.3   # threshold b = 1 + epsilon_budget

    # reward
    reward_mode: str = "risk_aware"      # "rate" or "risk_aware"
    lambda1: float = 1.0
    lambda2: float = 300.0

    # logging / eval
    log_every: int = 20
    validate_every: int = 50
    save_every: int = 200
    n_val_routes: Optional[int] = 50
    n_test_routes: Optional[int] = None

    # device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


cfg = Config()


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
    log_path = output_dir / f"train_{timestamp}.log"

    logger = logging.getLogger("recurrent_ppo_policy_dominant")
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
    """
    z: [B]
    return: [B, A], 1 valid / 0 invalid
    """
    B = len(z)
    mask = np.ones((B, len(action_sizes)), dtype=np.float32)
    restricted = z >= threshold_b
    if restricted.any():
        single_only = (action_sizes == 1).astype(np.float32)
        mask[restricted] = single_only[None, :]
    return mask


def masked_categorical_from_logits(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    masked_logits = logits.masked_fill(mask <= 0.0, -1e9)
    return Categorical(logits=masked_logits)


# ============================================================
# Dataset loading and split
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


def list_route_files(dataset_dir: str | Path) -> List[Path]:
    dataset_dir = Path(dataset_dir)
    files = sorted(dataset_dir.glob("*_result.npz"))
    if not files:
        raise RuntimeError(f"No *_result.npz files found in: {dataset_dir}")
    return files


def split_route_files(
    files: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Path]]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    files = list(files)
    rng = random.Random(seed)
    rng.shuffle(files)

    n = len(files)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        raise ValueError("Split produced an empty subset. Adjust ratios or dataset size.")

    return {
        "train": files[:n_train],
        "val": files[n_train:n_train + n_val],
        "test": files[n_train + n_val:],
    }


def inspect_dataset_shape(files: List[Path]) -> Tuple[int, int, int]:
    sample = load_route_capacity(files[0])
    T, n_cell, n_arr = sample.shape
    return T, n_cell, n_arr


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
# Training segment batch env
# ============================================================

class SegmentBatchEnv:
    """
    Each env samples:
      - one route
      - one start index
      - one segment of total length (burn_in + train_horizon)

    Route shape: [T, N_cell, N_arr]
    Action: choose 1 or 2 arrays
    Observation after action: all (cell, arr) values for selected arrays
    Reward: max observed rate, or risk-aware version
    """
    def __init__(
        self,
        routes: List[np.ndarray],
        num_envs: int,
        burn_in: int,
        horizon: int,
        action_list: Sequence[Tuple[int, ...]],
        alpha_risk: float,
        epsilon_budget: float,
        reward_mode: str,
        lambda1: float,
        lambda2: float,
        seed: int,
    ):
        self.routes = routes
        self.num_envs = num_envs
        self.burn_in = burn_in
        self.horizon = horizon
        self.total_len = burn_in + horizon
        self.action_list = list(action_list)
        self.alpha_risk = alpha_risk
        self.threshold_b = 1.0 + epsilon_budget
        self.reward_mode = reward_mode
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.rng = np.random.default_rng(seed)

        self.n_routes = len(routes)
        self.T, self.n_cell, self.n_arr = routes[0].shape

        if self.total_len > self.T:
            raise ValueError(
                f"burn_in + horizon = {self.total_len} exceeds route length T = {self.T}"
            )

        self.route_id = np.zeros(num_envs, dtype=np.int64)
        self.start = np.zeros(num_envs, dtype=np.int64)
        self.local_t = np.zeros(num_envs, dtype=np.int64)
        self.z = np.ones(num_envs, dtype=np.float32)
        self.done = np.zeros(num_envs, dtype=bool)

        self.reset_all()

    def reset_one(self, env_i: int) -> None:
        rid = int(self.rng.integers(0, self.n_routes))
        s = int(self.rng.integers(0, self.T - self.total_len + 1))
        self.route_id[env_i] = rid
        self.start[env_i] = s
        self.local_t[env_i] = 0
        self.z[env_i] = 1.0
        self.done[env_i] = False

    def reset_all(self) -> None:
        for i in range(self.num_envs):
            self.reset_one(i)

    def get_action_mask(self, action_sizes: np.ndarray) -> np.ndarray:
        return build_action_mask_from_z(self.z, action_sizes, self.threshold_b)

    def step(
        self,
        action_idx: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Returns:
            next_obs_values: [B, N_cell * N_arr]
            next_obs_mask  : [B, N_cell * N_arr]
            rewards        : [B]
            info           : dict
        """
        B = self.num_envs
        next_obs_values = np.zeros((B, self.n_cell * self.n_arr), dtype=np.float32)
        next_obs_mask = np.zeros((B, self.n_cell * self.n_arr), dtype=np.float32)
        rewards = np.zeros(B, dtype=np.float32)

        chosen_rate = np.zeros(B, dtype=np.float32)
        global_best_rate = np.zeros(B, dtype=np.float32)
        action_size = np.zeros(B, dtype=np.int64)
        restricted = np.zeros(B, dtype=np.float32)

        for i in range(B):
            if self.done[i]:
                continue

            rid = self.route_id[i]
            t = int(self.start[i] + self.local_t[i])

            route = self.routes[rid]
            rates_t = route[t]                # [N_cell, N_arr]
            global_best_rate[i] = float(rates_t.max())

            a_tuple = self.action_list[int(action_idx[i])]
            action_size[i] = len(a_tuple)
            restricted[i] = float(self.z[i] >= self.threshold_b)

            obs_vals = []
            for arr in a_tuple:
                for cell in range(self.n_cell):
                    flat_idx = cell * self.n_arr + arr
                    v = float(rates_t[cell, arr])
                    next_obs_values[i, flat_idx] = v
                    next_obs_mask[i, flat_idx] = 1.0
                    obs_vals.append(v)

            chosen_rate[i] = float(max(obs_vals)) if len(obs_vals) > 0 else 0.0

            if self.reward_mode == "rate":
                rewards[i] = chosen_rate[i]
            elif self.reward_mode == "risk_aware":
                penalty = self.lambda1 * (self.z[i] - 1.0) * (
                    chosen_rate[i] / (self.lambda2 + chosen_rate[i] + 1e-8)
                )
                rewards[i] = chosen_rate[i] * math.exp(-penalty)
            else:
                raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

            self.z[i] = (1.0 - self.alpha_risk) * self.z[i] + self.alpha_risk * float(len(a_tuple))

            self.local_t[i] += 1
            if self.local_t[i] >= self.total_len:
                self.done[i] = True

        info = {
            "chosen_rate": chosen_rate,
            "global_best_rate": global_best_rate,
            "z": self.z.copy(),
            "action_size": action_size,
            "restricted": restricted,
            "done": self.done.copy().astype(np.float32),
        }
        return next_obs_values, next_obs_mask, rewards, info


# ============================================================
# Evaluation on full route (sampled, no greedy)
# ============================================================

class FullRouteEvaluator:
    def __init__(
        self,
        routes: List[np.ndarray],
        action_list: Sequence[Tuple[int, ...]],
        alpha_risk: float,
        epsilon_budget: float,
        reward_mode: str,
        lambda1: float,
        lambda2: float,
    ):
        self.routes = routes
        self.action_list = list(action_list)
        self.alpha_risk = alpha_risk
        self.threshold_b = 1.0 + epsilon_budget
        self.reward_mode = reward_mode
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.T, self.n_cell, self.n_arr = routes[0].shape

    @torch.no_grad()
    def evaluate(
        self,
        model: "RecurrentActorCritic",
        device: torch.device,
        action_sizes: np.ndarray,
        max_routes: Optional[int] = None,
    ) -> Dict[str, float]:
        eval_routes = self.routes if max_routes is None else self.routes[:max_routes]

        total_return = []
        total_avg_reward = []
        total_avg_chosen_rate = []
        total_avg_best_rate = []
        total_best_hit = []
        total_pair_ratio = []
        total_restricted_ratio = []
        total_z_mean = []

        for route in tqdm(eval_routes, desc="Val-Sampled", leave=False):
            z = 1.0
            h = model.init_hidden(batch_size=1, device=device)

            prev_obs_values = np.zeros((1, self.n_cell * self.n_arr), dtype=np.float32)
            prev_obs_mask = np.zeros((1, self.n_cell * self.n_arr), dtype=np.float32)

            rewards = []
            chosen_rates = []
            global_best_rates = []
            pair_flags = []
            restricted_flags = []
            z_trace = []

            for t in range(self.T):
                obs_t = build_obs_input(
                    prev_obs_values,
                    prev_obs_mask,
                    np.asarray([z], dtype=np.float32),
                )
                obs_tensor = torch.from_numpy(obs_t).to(device)

                logits, _, _, h = model.forward_step(obs_tensor, h)

                mask_np = build_action_mask_from_z(
                    np.asarray([z], dtype=np.float32),
                    action_sizes,
                    self.threshold_b,
                )
                mask_t = torch.from_numpy(mask_np).to(device)

                dist = masked_categorical_from_logits(logits, mask_t)
                action_idx = int(dist.sample().item())

                a_tuple = self.action_list[action_idx]
                rates_t = route[t]
                global_best = float(rates_t.max())

                obs_values = np.zeros((1, self.n_cell * self.n_arr), dtype=np.float32)
                obs_mask = np.zeros((1, self.n_cell * self.n_arr), dtype=np.float32)
                observed_vals = []

                for arr in a_tuple:
                    for cell in range(self.n_cell):
                        flat_idx = cell * self.n_arr + arr
                        v = float(rates_t[cell, arr])
                        obs_values[0, flat_idx] = v
                        obs_mask[0, flat_idx] = 1.0
                        observed_vals.append(v)

                chosen = float(max(observed_vals)) if len(observed_vals) > 0 else 0.0

                if self.reward_mode == "rate":
                    reward = chosen
                elif self.reward_mode == "risk_aware":
                    penalty = self.lambda1 * (z - 1.0) * (
                        chosen / (self.lambda2 + chosen + 1e-8)
                    )
                    reward = chosen * math.exp(-penalty)
                else:
                    raise ValueError(f"Unknown reward_mode: {self.reward_mode}")

                rewards.append(reward)
                chosen_rates.append(chosen)
                global_best_rates.append(global_best)
                pair_flags.append(1.0 if len(a_tuple) == 2 else 0.0)
                restricted_flags.append(1.0 if z >= self.threshold_b else 0.0)
                z_trace.append(z)

                z = (1.0 - self.alpha_risk) * z + self.alpha_risk * float(len(a_tuple))
                prev_obs_values = obs_values
                prev_obs_mask = obs_mask

            rewards = np.asarray(rewards, dtype=np.float32)
            chosen_rates = np.asarray(chosen_rates, dtype=np.float32)
            global_best_rates = np.asarray(global_best_rates, dtype=np.float32)
            pair_flags = np.asarray(pair_flags, dtype=np.float32)
            restricted_flags = np.asarray(restricted_flags, dtype=np.float32)
            z_trace = np.asarray(z_trace, dtype=np.float32)

            avg_chosen_rate = float(chosen_rates.mean())
            avg_best_rate = float(global_best_rates.mean())
            hit = float(np.mean(np.isclose(chosen_rates, global_best_rates, atol=1e-6)))

            total_return.append(float(rewards.sum()))
            total_avg_reward.append(float(rewards.mean()))
            total_avg_chosen_rate.append(avg_chosen_rate)
            total_avg_best_rate.append(avg_best_rate)
            total_best_hit.append(hit)
            total_pair_ratio.append(float(pair_flags.mean()))
            total_restricted_ratio.append(float(restricted_flags.mean()))
            total_z_mean.append(float(z_trace.mean()))

        mean_chosen_rate = float(np.mean(total_avg_chosen_rate))
        mean_best_rate = float(np.mean(total_avg_best_rate))
        best_ratio = float(mean_chosen_rate / (mean_best_rate + 1e-8))

        return {
            "num_routes": len(eval_routes),
            "mean_return": float(np.mean(total_return)),
            "mean_reward_per_step": float(np.mean(total_avg_reward)),
            "mean_chosen_rate": mean_chosen_rate,
            "mean_best_rate": mean_best_rate,
            "mean_hit_global_best_rate": float(np.mean(total_best_hit)),
            "best_ratio": best_ratio,
            "pair_ratio": float(np.mean(total_pair_ratio)),
            "single_ratio": 1.0 - float(np.mean(total_pair_ratio)),
            "z_mean": float(np.mean(total_z_mean)),
            "restricted_ratio": float(np.mean(total_restricted_ratio)),
        }


# ============================================================
# Model (policy dominant)
# ============================================================

class RecurrentActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        gru_hidden_dim: int,
        num_actions: int,
        obs_out_dim: int,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.policy_head = nn.Linear(gru_hidden_dim, num_actions)
        self.value_head = nn.Linear(gru_hidden_dim, 1)
        self.obs_head = nn.Linear(gru_hidden_dim, obs_out_dim)

        self.gru_hidden_dim = gru_hidden_dim
        self.obs_out_dim = obs_out_dim

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)

    def forward(
        self,
        obs_seq: torch.Tensor,   # [B, T, D]
        h0: torch.Tensor,        # [1, B, H]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.encoder(obs_seq)
        out, hN = self.gru(x, h0)

        # policy dominates shared backbone
        logits = self.policy_head(out)

        # auxiliary heads do not backprop into backbone
        out_detached = out.detach()
        values = self.value_head(out_detached).squeeze(-1)
        obs_pred = self.obs_head(out_detached)

        return logits, values, obs_pred, hN

    def forward_step(
        self,
        obs_t: torch.Tensor,     # [B, D]
        h: torch.Tensor,         # [1, B, H]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, obs_pred, hN = self.forward(obs_t.unsqueeze(1), h)
        return logits[:, 0], values[:, 0], obs_pred[:, 0], hN


# ============================================================
# Rollout buffer
# ============================================================

class RolloutBuffer:
    def __init__(self, num_envs: int, horizon: int, obs_dim: int, hidden_dim: int, obs_out_dim: int):
        self.num_envs = num_envs
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.obs_out_dim = obs_out_dim
        self.reset()

    def reset(self) -> None:
        B, H, D, G, O = self.num_envs, self.horizon, self.obs_dim, self.hidden_dim, self.obs_out_dim

        self.obs = np.zeros((B, H, D), dtype=np.float32)
        self.actions = np.zeros((B, H), dtype=np.int64)
        self.old_logp = np.zeros((B, H), dtype=np.float32)
        self.rewards = np.zeros((B, H), dtype=np.float32)
        self.values = np.zeros((B, H), dtype=np.float32)
        self.dones = np.zeros((B, H), dtype=np.float32)
        self.z = np.zeros((B, H), dtype=np.float32)
        self.action_sizes = np.zeros((B, H), dtype=np.int64)
        self.chosen_rates = np.zeros((B, H), dtype=np.float32)
        self.global_best_rates = np.zeros((B, H), dtype=np.float32)
        self.restricted = np.zeros((B, H), dtype=np.float32)
        self.h0 = np.zeros((B, 1, G), dtype=np.float32)

        self.obs_target = np.zeros((B, H, O), dtype=np.float32)
        self.obs_target_mask = np.zeros((B, H, O), dtype=np.float32)

        self.advantages = np.zeros((B, H), dtype=np.float32)
        self.returns = np.zeros((B, H), dtype=np.float32)

    def compute_gae(self, last_values: np.ndarray, gamma: float, gae_lambda: float) -> None:
        B, H = self.rewards.shape
        adv = np.zeros(B, dtype=np.float32)

        for t in reversed(range(H)):
            next_value = last_values if t == H - 1 else self.values[:, t + 1]
            next_nonterminal = 1.0 - self.dones[:, t]
            delta = self.rewards[:, t] + gamma * next_value * next_nonterminal - self.values[:, t]
            adv = delta + gamma * gae_lambda * next_nonterminal * adv
            self.advantages[:, t] = adv

        self.returns = self.advantages + self.values


# ============================================================
# Train helpers
# ============================================================

def build_obs_input(
    obs_values: np.ndarray,
    obs_mask: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    return np.concatenate(
        [obs_values, obs_mask, z[:, None].astype(np.float32)],
        axis=1,
    ).astype(np.float32)


@torch.no_grad()
def run_burn_in(
    env: SegmentBatchEnv,
    model: RecurrentActorCritic,
    device: torch.device,
    action_sizes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
    B = env.num_envs
    obs_values = np.zeros((B, env.n_cell * env.n_arr), dtype=np.float32)
    obs_mask = np.zeros((B, env.n_cell * env.n_arr), dtype=np.float32)

    h = model.init_hidden(batch_size=B, device=device)

    for _ in range(env.burn_in):
        obs_input = build_obs_input(obs_values, obs_mask, env.z)
        obs_t = torch.from_numpy(obs_input).to(device)

        logits, _, _, h = model.forward_step(obs_t, h)

        mask_np = env.get_action_mask(action_sizes)
        mask_t = torch.from_numpy(mask_np).to(device)

        dist = masked_categorical_from_logits(logits, mask_t)
        action_idx = dist.sample().cpu().numpy()

        obs_values, obs_mask, _, _ = env.step(action_idx)

    return obs_values, obs_mask, h


@torch.no_grad()
def collect_rollout(
    env: SegmentBatchEnv,
    model: RecurrentActorCritic,
    device: torch.device,
    action_sizes: np.ndarray,
    cfg: Config,
) -> RolloutBuffer:
    env.reset_all()
    obs_values, obs_mask, h = run_burn_in(env, model, device, action_sizes)

    B = env.num_envs
    obs_dim = 2 * env.n_cell * env.n_arr + 1
    obs_out_dim = env.n_cell * env.n_arr

    buffer = RolloutBuffer(B, env.horizon, obs_dim, model.gru_hidden_dim, obs_out_dim)
    buffer.h0[:, 0, :] = h[0].cpu().numpy()

    for t in range(env.horizon):
        obs_input = build_obs_input(obs_values, obs_mask, env.z)
        obs_t = torch.from_numpy(obs_input).to(device)

        logits, values, _, h = model.forward_step(obs_t, h)

        mask_np = env.get_action_mask(action_sizes)
        mask_t = torch.from_numpy(mask_np).to(device)

        dist = masked_categorical_from_logits(logits, mask_t)
        action_t = dist.sample()
        logp_t = dist.log_prob(action_t)

        action_np = action_t.cpu().numpy()
        logp_np = logp_t.cpu().numpy()
        values_np = values.cpu().numpy()

        next_obs_values, next_obs_mask, rewards, info = env.step(action_np)

        buffer.obs[:, t] = obs_input
        buffer.actions[:, t] = action_np
        buffer.old_logp[:, t] = logp_np
        buffer.rewards[:, t] = rewards
        buffer.values[:, t] = values_np
        buffer.dones[:, t] = info["done"]
        buffer.z[:, t] = info["z"]
        buffer.action_sizes[:, t] = info["action_size"]
        buffer.chosen_rates[:, t] = info["chosen_rate"]
        buffer.global_best_rates[:, t] = info["global_best_rate"]
        buffer.restricted[:, t] = info["restricted"]

        buffer.obs_target[:, t] = next_obs_values
        buffer.obs_target_mask[:, t] = next_obs_mask

        obs_values = next_obs_values
        obs_mask = next_obs_mask

    obs_input = build_obs_input(obs_values, obs_mask, env.z)
    obs_t = torch.from_numpy(obs_input).to(device)
    _, last_values_t, _, _ = model.forward_step(obs_t, h)
    last_values = last_values_t.cpu().numpy()

    buffer.compute_gae(last_values, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
    return buffer


def ppo_update(
    model: RecurrentActorCritic,
    optimizer: optim.Optimizer,
    buffer: RolloutBuffer,
    device: torch.device,
    action_sizes: np.ndarray,
    threshold_b: float,
    cfg: Config,
) -> Dict[str, float]:
    B, H, _ = buffer.obs.shape
    all_idx = np.arange(B)

    adv = buffer.advantages.copy()
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    stats = {
        "pi_loss": [],
        "v_loss": [],
        "entropy": [],
        "obs_loss": [],
        "total_loss": [],
    }

    for _ in range(cfg.update_epochs):
        np.random.shuffle(all_idx)

        for start in range(0, B, cfg.minibatch_size):
            mb_idx = all_idx[start:start + cfg.minibatch_size]
            if len(mb_idx) == 0:
                continue

            obs = torch.from_numpy(buffer.obs[mb_idx]).to(device)
            actions = torch.from_numpy(buffer.actions[mb_idx]).to(device)
            old_logp = torch.from_numpy(buffer.old_logp[mb_idx]).to(device)
            returns = torch.from_numpy(buffer.returns[mb_idx]).to(device)
            advantages = torch.from_numpy(adv[mb_idx]).to(device)
            z = buffer.z[mb_idx]
            h0 = torch.from_numpy(buffer.h0[mb_idx].transpose(1, 0, 2)).to(device)

            obs_target = torch.from_numpy(buffer.obs_target[mb_idx]).to(device)
            obs_target_mask = torch.from_numpy(buffer.obs_target_mask[mb_idx]).to(device)

            logits, values, obs_pred, _ = model(obs, h0)

            mask_np = np.stack(
                [build_action_mask_from_z(z[:, t], action_sizes, threshold_b) for t in range(H)],
                axis=1,
            )
            mask = torch.from_numpy(mask_np).to(device)

            dist = masked_categorical_from_logits(logits, mask)
            new_logp = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
            pi_loss = -torch.min(surr1, surr2).mean()

            v_loss = ((values - returns) ** 2).mean()

            obs_sqerr = (obs_pred - obs_target) ** 2
            obs_loss = (obs_sqerr * obs_target_mask).sum() / (obs_target_mask.sum() + 1e-8)

            total_loss = (
                pi_loss
                + cfg.value_coef * v_loss
                - cfg.entropy_coef * entropy
                + cfg.obs_loss_coef * obs_loss
            )

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()

            stats["pi_loss"].append(float(pi_loss.item()))
            stats["v_loss"].append(float(v_loss.item()))
            stats["entropy"].append(float(entropy.item()))
            stats["obs_loss"].append(float(obs_loss.item()))
            stats["total_loss"].append(float(total_loss.item()))

    return {k: float(np.mean(v)) for k, v in stats.items()}


def summarize_rollout(buffer: RolloutBuffer) -> Dict[str, float]:
    chosen = buffer.chosen_rates
    best = buffer.global_best_rates
    pair = (buffer.action_sizes == 2).astype(np.float32)
    hit = np.isclose(chosen, best, atol=1e-6).astype(np.float32)

    avg_ret = float(buffer.rewards.mean())
    avg_chosen_rate = float(chosen.mean())
    avg_best_rate = float(best.mean())
    best_ratio = float(avg_chosen_rate / (avg_best_rate + 1e-8))

    return {
        "avg_ret": avg_ret,
        "avg_chosen_rate": avg_chosen_rate,
        "avg_best_rate": avg_best_rate,
        "best_hit_rate": float(hit.mean()),
        "best_ratio": best_ratio,
        "single": float(1.0 - pair.mean()),
        "pair": float(pair.mean()),
        "z_mean": float(buffer.z.mean()),
        "restricted": float(buffer.restricted.mean()),
    }


def save_checkpoint(
    path: Path,
    model: RecurrentActorCritic,
    optimizer: optim.Optimizer,
    cfg: Config,
    extra: Dict[str, float],
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
        "extra": extra,
    }
    torch.save(payload, path)


# ============================================================
# Main
# ============================================================

def main() -> None:
    set_seed(cfg.seed)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"PPO_alpha_{cfg.alpha_risk}_eps_{cfg.epsilon_budget}_h{cfg.train_horizon}_lr{cfg.lr}"
    output_dir = Path(cfg.output_dir) / f"{exp_name}_{run_id}"
    ensure_dir(output_dir)

    logger = setup_logger(output_dir)
    writer = SummaryWriter(log_dir=str(output_dir / "tb"))

    files = list_route_files(cfg.dataset_dir)
    split = split_route_files(
        files=files,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
    )

    T, n_cell, n_arr = inspect_dataset_shape(files)
    logger.info(f"Dataset shape: T={T}, N_cell={n_cell}, N_arr={n_arr}")
    logger.info(
        f"Split sizes: train={len(split['train'])}, "
        f"val={len(split['val'])}, test={len(split['test'])}"
    )

    logger.info("Loading route files into memory...")
    train_routes = load_all_routes(split["train"], (T, n_cell, n_arr))
    val_routes = load_all_routes(split["val"], (T, n_cell, n_arr))
    test_routes = load_all_routes(split["test"], (T, n_cell, n_arr))
    logger.info("All routes loaded.")

    action_list = build_action_list(n_arr)
    action_sizes = build_action_sizes(action_list)
    num_actions = len(action_list)
    threshold_b = 1.0 + cfg.epsilon_budget

    obs_dim = 2 * n_cell * n_arr + 1
    obs_out_dim = n_cell * n_arr
    device = torch.device(cfg.device)

    model = RecurrentActorCritic(
        obs_dim=obs_dim,
        hidden_dim=cfg.encoder_hidden_dim,
        gru_hidden_dim=cfg.gru_hidden_dim,
        num_actions=num_actions,
        obs_out_dim=obs_out_dim,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    train_env = SegmentBatchEnv(
        routes=train_routes,
        num_envs=cfg.num_envs,
        burn_in=cfg.burn_in,
        horizon=cfg.train_horizon,
        action_list=action_list,
        alpha_risk=cfg.alpha_risk,
        epsilon_budget=cfg.epsilon_budget,
        reward_mode=cfg.reward_mode,
        lambda1=cfg.lambda1,
        lambda2=cfg.lambda2,
        seed=cfg.seed,
    )

    val_eval = FullRouteEvaluator(
        routes=val_routes,
        action_list=action_list,
        alpha_risk=cfg.alpha_risk,
        epsilon_budget=cfg.epsilon_budget,
        reward_mode=cfg.reward_mode,
        lambda1=cfg.lambda1,
        lambda2=cfg.lambda2,
    )

    test_eval = FullRouteEvaluator(
        routes=test_routes,
        action_list=action_list,
        alpha_risk=cfg.alpha_risk,
        epsilon_budget=cfg.epsilon_budget,
        reward_mode=cfg.reward_mode,
        lambda1=cfg.lambda1,
        lambda2=cfg.lambda2,
    )

    config_to_save = asdict(cfg).copy()
    config_to_save["split_files"] = {
        "train": [str(p) for p in split["train"]],
        "val": [str(p) for p in split["val"]],
        "test": [str(p) for p in split["test"]],
    }

    with open(output_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_to_save, f, indent=2)

    logger.info("Config:")
    logger.info(json.dumps(config_to_save, indent=2))

    best_val_ratio = -float("inf")
    t0 = time.time()

    logger.info("Start training...")
    pbar = tqdm(range(1, cfg.total_updates + 1), desc="Training")

    for update in pbar:
        model.train()

        buffer = collect_rollout(
            env=train_env,
            model=model,
            device=device,
            action_sizes=action_sizes,
            cfg=cfg,
        )

        train_stats = summarize_rollout(buffer)

        loss_stats = ppo_update(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            device=device,
            action_sizes=action_sizes,
            threshold_b=threshold_b,
            cfg=cfg,
        )

        global_step = update * cfg.num_envs * cfg.train_horizon

        pbar.set_postfix({
            "rate": f"{train_stats['avg_chosen_rate']:.1f}",
            "ratio": f"{train_stats['best_ratio']:.3f}",
            "hit": f"{train_stats['best_hit_rate']:.3f}",
            "pair": f"{train_stats['pair']:.2f}",
            "z": f"{train_stats['z_mean']:.2f}",
            "pi": f"{loss_stats['pi_loss']:.3f}",
            "v": f"{loss_stats['v_loss']:.1e}",
            "obs": f"{loss_stats['obs_loss']:.3f}",
        })

        if update % cfg.log_every == 0 or update == 1:
            elapsed_min = (time.time() - t0) / 60.0

            logger.info(
                f"[Rollout {update}/{cfg.total_updates}] "
                f"step={global_step} | "
                f"avg_ret={train_stats['avg_ret']:.4f} | "
                f"avg_chosen_rate={train_stats['avg_chosen_rate']:.4f} | "
                f"best_hit_rate={train_stats['best_hit_rate']:.4f} | "
                f"best_ratio={train_stats['best_ratio']:.4f} | "
                f"single={train_stats['single']:.3f} | "
                f"pair={train_stats['pair']:.3f} | "
                f"z_mean={train_stats['z_mean']:.3f} | "
                f"restricted={train_stats['restricted']:.3f} | "
                f"pi_loss={loss_stats['pi_loss']:.4f} | "
                f"v_loss={loss_stats['v_loss']:.4f} | "
                f"entropy={loss_stats['entropy']:.4f} | "
                f"obs_loss={loss_stats['obs_loss']:.4f} | "
                f"total_loss={loss_stats['total_loss']:.4f} | "
                f"elapsed={elapsed_min:.1f} min"
            )

            writer.add_scalar("train/avg_ret", train_stats["avg_ret"], global_step)
            writer.add_scalar("train/avg_chosen_rate", train_stats["avg_chosen_rate"], global_step)
            writer.add_scalar("train/best_hit_rate", train_stats["best_hit_rate"], global_step)
            writer.add_scalar("train/best_ratio", train_stats["best_ratio"], global_step)
            writer.add_scalar("train/single_ratio", train_stats["single"], global_step)
            writer.add_scalar("train/pair_ratio", train_stats["pair"], global_step)
            writer.add_scalar("train/z_mean", train_stats["z_mean"], global_step)
            writer.add_scalar("train/restricted_ratio", train_stats["restricted"], global_step)

            writer.add_scalar("loss/pi_loss", loss_stats["pi_loss"], global_step)
            writer.add_scalar("loss/v_loss", loss_stats["v_loss"], global_step)
            writer.add_scalar("loss/entropy", loss_stats["entropy"], global_step)
            writer.add_scalar("loss/obs_loss", loss_stats["obs_loss"], global_step)
            writer.add_scalar("loss/total_loss", loss_stats["total_loss"], global_step)

        if update % cfg.validate_every == 0 or update == cfg.total_updates:
            model.eval()
            val_stats = val_eval.evaluate(
                model=model,
                device=device,
                action_sizes=action_sizes,
                max_routes=cfg.n_val_routes,
            )

            logger.info(
                f"[VAL-SAMPLED] routes={val_stats['num_routes']} | "
                f"avg_return={val_stats['mean_return']:.4f} | "
                f"avg_reward_per_step={val_stats['mean_reward_per_step']:.4f} | "
                f"avg_chosen_rate={val_stats['mean_chosen_rate']:.4f} | "
                f"avg_hit_global_best_rate={val_stats['mean_hit_global_best_rate']:.4f} | "
                f"best_ratio={val_stats['best_ratio']:.4f} | "
                f"single={val_stats['single_ratio']:.3f} | "
                f"pair={val_stats['pair_ratio']:.3f} | "
                f"z_mean={val_stats['z_mean']:.3f} | "
                f"restricted={val_stats['restricted_ratio']:.3f}"
            )

            writer.add_scalar("val_sampled/mean_return", val_stats["mean_return"], global_step)
            writer.add_scalar("val_sampled/mean_reward_per_step", val_stats["mean_reward_per_step"], global_step)
            writer.add_scalar("val_sampled/mean_chosen_rate", val_stats["mean_chosen_rate"], global_step)
            writer.add_scalar("val_sampled/mean_best_rate", val_stats["mean_best_rate"], global_step)
            writer.add_scalar("val_sampled/mean_hit_global_best_rate", val_stats["mean_hit_global_best_rate"], global_step)
            writer.add_scalar("val_sampled/best_ratio", val_stats["best_ratio"], global_step)
            writer.add_scalar("val_sampled/single_ratio", val_stats["single_ratio"], global_step)
            writer.add_scalar("val_sampled/pair_ratio", val_stats["pair_ratio"], global_step)
            writer.add_scalar("val_sampled/z_mean", val_stats["z_mean"], global_step)
            writer.add_scalar("val_sampled/restricted_ratio", val_stats["restricted_ratio"], global_step)

            if val_stats["best_ratio"] > best_val_ratio:
                best_val_ratio = val_stats["best_ratio"]
                save_checkpoint(
                    output_dir / "best.pt",
                    model,
                    optimizer,
                    cfg,
                    {
                        "best_val_ratio": best_val_ratio,
                        "update": update,
                        "mean_chosen_rate": val_stats["mean_chosen_rate"],
                        "mean_hit_global_best_rate": val_stats["mean_hit_global_best_rate"],
                    },
                )
                logger.info(f"Saved new best checkpoint by best_ratio={best_val_ratio:.4f}")

        if update % cfg.save_every == 0:
            save_checkpoint(
                output_dir / f"ckpt_update_{update}.pt",
                model,
                optimizer,
                cfg,
                {"update": update},
            )
            logger.info(f"Saved checkpoint: ckpt_update_{update}.pt")

    save_checkpoint(
        output_dir / "last.pt",
        model,
        optimizer,
        cfg,
        {"best_val_ratio": best_val_ratio, "update": cfg.total_updates},
    )

    logger.info("Training done.")

    model.eval()

    logger.info("=== TEST-SAMPLED (last checkpoint) ===")
    last_test_stats = test_eval.evaluate(
        model=model,
        device=device,
        action_sizes=action_sizes,
        max_routes=cfg.n_test_routes,
    )
    logger.info(json.dumps(last_test_stats, indent=2))

    best_path = output_dir / "best.pt"
    if best_path.exists():
        payload = torch.load(best_path, map_location=device)
        model.load_state_dict(payload["model_state_dict"])

        logger.info("=== TEST-SAMPLED (best checkpoint) ===")
        best_test_stats = test_eval.evaluate(
            model=model,
            device=device,
            action_sizes=action_sizes,
            max_routes=cfg.n_test_routes,
        )
        logger.info(json.dumps(best_test_stats, indent=2))

    writer.close()


if __name__ == "__main__":
    main()