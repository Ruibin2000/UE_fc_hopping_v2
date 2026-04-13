from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple
import json
import logging
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.auto import tqdm


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensor(
    x: np.ndarray | Sequence[float],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.as_tensor(x, dtype=dtype, device=device)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, tuple):
            return list(obj)
        if isinstance(obj, list):
            return [_convert(x) for x in obj]
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_convert(data), f, indent=2)


def setup_logger(log_dir: Path, log_name: str = "train.log") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    logger = logging.getLogger("recurrent_ppo_risk")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def log_config_block(logger: logging.Logger, title: str, cfg: Dict[str, Any]) -> None:
    logger.info(f"=== {title} ===")
    for k, v in cfg.items():
        logger.info(f"{k:<28}: {v}")


def save_z_plot(
    z_values: List[float],
    epsilon: float,
    save_path: Path,
    title: str = "Risk state z over time",
) -> None:
    if len(z_values) == 0:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(z_values))
    threshold = 1.0 + epsilon

    plt.figure(figsize=(8, 4))
    plt.plot(x, z_values, label="z_t")
    plt.axhline(threshold, linestyle="--", label=f"threshold={threshold:.2f}")
    plt.xlabel("Step")
    plt.ylabel("z")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# Action space
# ============================================================

class SubsetActionSpace:
    """
    Valid action space:
        P_{<=2} = {A subset of {0,...,N_arr-1}: 1 <= |A| <= 2}
    """

    def __init__(self, n_arr: int) -> None:
        if n_arr < 1:
            raise ValueError("n_arr must be >= 1")

        self.n_arr = n_arr
        self.single_subsets: List[Tuple[int, ...]] = [(a,) for a in range(n_arr)]
        self.pair_subsets: List[Tuple[int, ...]] = [
            (a, b) for a in range(n_arr) for b in range(a + 1, n_arr)
        ]
        self.action_list: List[Tuple[int, ...]] = self.single_subsets + self.pair_subsets
        self.action_to_idx: Dict[Tuple[int, ...], int] = {
            subset: idx for idx, subset in enumerate(self.action_list)
        }

    @property
    def n_actions(self) -> int:
        return len(self.action_list)

    def idx_to_subset(self, action_idx: int) -> Tuple[int, ...]:
        return self.action_list[action_idx]

    def subset_to_idx(self, subset: Tuple[int, ...]) -> int:
        subset = tuple(sorted(subset))
        return self.action_to_idx[subset]


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

    train_files = files[:n_train]
    val_files = files[n_train:n_train + n_val]
    test_files = files[n_train + n_val:]

    return {
        "train": train_files,
        "val": val_files,
        "test": test_files,
    }


def inspect_dataset_shape(files: List[Path]) -> Tuple[int, int, int]:
    sample = load_route_capacity(files[0])
    T, n_cell, n_arr = sample.shape
    return T, n_cell, n_arr


# ============================================================
# Risk-controlled environment
# ============================================================

class RouteDatasetRiskEnv:
    """
    Dataset-driven environment using real routes.

    At each step:
    - action_idx -> subset of arrays
    - only channels on those arrays are observed
    - serving channel is selected automatically as the best observed feasible channel
    - reward = chosen_rate
    - risk state is updated:
          z_{t+1} = (1 - alpha) z_t + alpha * |A_t|

    Constraint:
        if z_t < 1 + epsilon:
            all single/pair actions allowed
        else:
            only single-array actions allowed
    """

    def __init__(
        self,
        route_files: List[Path],
        alpha: float,
        epsilon: float,
        episode_horizon: int,
        seed: int,
        training: bool,
        random_start: bool = True,
        cache_routes: bool = False,
    ) -> None:
        if len(route_files) == 0:
            raise ValueError("route_files must not be empty")

        self.route_files = list(route_files)
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.episode_horizon = int(episode_horizon)
        self.training = bool(training)
        self.random_start = bool(random_start)
        self.cache_routes = bool(cache_routes)

        self.rng = np.random.default_rng(seed)

        first = load_route_capacity(self.route_files[0])
        self.n_cell = first.shape[1]
        self.n_arr = first.shape[2]
        self.action_space = SubsetActionSpace(self.n_arr)

        self.route_cache: Dict[str, np.ndarray] = {}
        if self.cache_routes:
            for p in self.route_files:
                self.route_cache[str(p)] = load_route_capacity(p)

        self.current_route: Optional[np.ndarray] = None
        self.current_route_path: Optional[Path] = None
        self.current_t: int = 0
        self.current_start_t: int = 0
        self.steps_in_episode: int = 0

        self.prev_reward = 0.0
        self.prev_action_idx = 0

        self.last_observed_rate = np.zeros((self.n_cell, self.n_arr), dtype=np.float32)
        self.last_observed_mask = np.zeros((self.n_cell, self.n_arr), dtype=np.float32)
        self.last_observed_time = np.full((self.n_cell, self.n_arr), fill_value=-1, dtype=np.int32)

        self.z = 1.0

    @property
    def obs_dim(self) -> int:
        channels = self.n_cell * self.n_arr
        return channels * 3 + self.action_space.n_actions + 2

    def _load_route(self, path: Path) -> np.ndarray:
        if self.cache_routes:
            return self.route_cache[str(path)]
        return load_route_capacity(path)

    def _sample_route_and_start(self) -> Tuple[Path, int]:
        path = self.route_files[self.rng.integers(0, len(self.route_files))]
        route = self._load_route(path)
        T = route.shape[0]

        if self.training and self.random_start and T > self.episode_horizon:
            max_start = T - self.episode_horizon
            start_t = int(self.rng.integers(0, max_start + 1))
        else:
            start_t = 0

        return path, start_t

    def _get_action_mask(self) -> np.ndarray:
        b = 1.0 + self.epsilon
        mask = np.ones(self.action_space.n_actions, dtype=np.float32)

        if self.z >= b:
            for idx, subset in enumerate(self.action_space.action_list):
                if len(subset) == 2:
                    mask[idx] = 0.0

        return mask

    def _build_obs(self) -> np.ndarray:
        staleness = np.zeros((self.n_cell, self.n_arr), dtype=np.float32)
        never_seen = self.last_observed_time < 0
        staleness[never_seen] = float(self.episode_horizon)
        staleness[~never_seen] = (
            self.steps_in_episode - self.last_observed_time[~never_seen]
        ).astype(np.float32)

        prev_action_one_hot = np.zeros(self.action_space.n_actions, dtype=np.float32)
        prev_action_one_hot[self.prev_action_idx] = 1.0

        obs = np.concatenate(
            [
                self.last_observed_mask.reshape(-1),
                self.last_observed_rate.reshape(-1),
                staleness.reshape(-1),
                prev_action_one_hot,
                np.array([self.prev_reward], dtype=np.float32),
                np.array([self.z], dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

        return obs

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        path, start_t = self._sample_route_and_start()
        route = self._load_route(path)

        self.current_route = route
        self.current_route_path = path
        self.current_start_t = start_t
        self.current_t = start_t
        self.steps_in_episode = 0

        self.prev_reward = 0.0
        self.prev_action_idx = 0
        self.last_observed_rate.fill(0.0)
        self.last_observed_mask.fill(0.0)
        self.last_observed_time.fill(-1)
        self.z = 1.0

        obs = self._build_obs()
        info = {
            "route_file": path.name,
            "route_length": int(route.shape[0]),
            "start_t": int(start_t),
            "z": float(self.z),
            "action_mask": self._get_action_mask(),
        }
        return obs, info

    def step(self, action_idx: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_route is None:
            raise RuntimeError("Call reset() before step()")

        action_mask = self._get_action_mask()
        if action_mask[action_idx] <= 0:
            raise ValueError(f"Masked action selected: {action_idx}, z={self.z:.6f}")

        subset = self.action_space.idx_to_subset(action_idx)

        rates = self.current_route[self.current_t]  # [N_cell, N_arr]
        observed = np.full_like(rates, fill_value=np.nan)

        for arr_idx in subset:
            observed[:, arr_idx] = rates[:, arr_idx]

        observed_mask = np.isfinite(observed)

        updated_rates = np.where(observed_mask, observed, self.last_observed_rate)
        self.last_observed_rate = updated_rates.astype(np.float32)
        self.last_observed_mask = observed_mask.astype(np.float32)
        self.last_observed_time[observed_mask] = self.steps_in_episode

        if np.isfinite(observed).any():
            chosen_rate = float(np.nanmax(observed))
            serving_flat_idx = int(np.nanargmax(observed))
            serving_cell, serving_arr = np.unravel_index(serving_flat_idx, observed.shape)
            serving_channel = (int(serving_cell), int(serving_arr))
        else:
            chosen_rate = 0.0
            serving_channel = None

        reward = chosen_rate

        global_best_rate = float(np.max(rates))
        global_best_flat_idx = int(np.argmax(rates))
        global_best_cell, global_best_arr = np.unravel_index(global_best_flat_idx, rates.shape)
        global_best_channel = (int(global_best_cell), int(global_best_arr))

        hit_global_best = bool(
            np.isclose(chosen_rate, global_best_rate, rtol=1e-6, atol=1e-8)
        )

        z_before = float(self.z)
        self.z = (1.0 - self.alpha) * self.z + self.alpha * len(subset)
        z_after = float(self.z)

        self.prev_reward = float(reward)
        self.prev_action_idx = action_idx

        self.current_t += 1
        self.steps_in_episode += 1

        route_done = self.current_t >= self.current_route.shape[0]
        horizon_done = self.steps_in_episode >= self.episode_horizon
        done = bool(route_done or horizon_done)
        truncated = False

        next_obs = self._build_obs()
        next_mask = self._get_action_mask()

        info = {
            "route_file": self.current_route_path.name if self.current_route_path else None,
            "subset": subset,
            "chosen_rate": chosen_rate,
            "serving_channel": serving_channel,
            "global_best_rate": global_best_rate,
            "global_best_channel": global_best_channel,
            "hit_global_best": hit_global_best,
            "time_index": int(self.current_t - 1),
            "z_before": z_before,
            "z_after": z_after,
            "action_mask": next_mask,
        }
        return next_obs, float(reward), done, truncated, info


# ============================================================
# Model
# ============================================================

class RecurrentPPONet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        encoder_hidden_dim: int = 128,
        gru_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.encoder_hidden_dim = encoder_hidden_dim
        self.gru_hidden_dim = gru_hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, encoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(encoder_hidden_dim, encoder_hidden_dim),
            nn.ReLU(),
        )

        self.gru = nn.GRU(
            input_size=encoder_hidden_dim,
            hidden_size=gru_hidden_dim,
            num_layers=1,
            batch_first=False,
        )

        self.policy_head = nn.Linear(gru_hidden_dim, n_actions)
        self.value_head = nn.Linear(gru_hidden_dim, 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(1, batch_size, self.gru_hidden_dim, device=device)

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        h0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T, B, _ = obs_seq.shape
        flat = obs_seq.reshape(T * B, self.obs_dim)
        enc = self.encoder(flat).reshape(T, B, self.encoder_hidden_dim)
        out, hT = self.gru(enc, h0)
        logits = self.policy_head(out)
        values = self.value_head(out).squeeze(-1)
        return logits, values, hT

    @staticmethod
    def apply_action_mask(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        invalid = action_mask <= 0
        return logits.masked_fill(invalid, -1e9)

    def act(
        self,
        obs: torch.Tensor,
        h: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        obs_seq = obs.unsqueeze(0)
        logits, values, next_hidden = self.forward_sequence(obs_seq, h)
        logits = logits[0, 0]
        value = values[0, 0]

        masked_logits = self.apply_action_mask(logits, action_mask[0])
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.unsqueeze(0), log_prob.unsqueeze(0), value.unsqueeze(0), next_hidden

    def evaluate_actions_sequence(
        self,
        obs_seq: torch.Tensor,
        h0: torch.Tensor,
        action_seq: torch.Tensor,
        action_mask_seq: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values, _ = self.forward_sequence(obs_seq, h0)
        masked_logits = self.apply_action_mask(logits, action_mask_seq)
        dist = Categorical(logits=masked_logits)
        log_probs = dist.log_prob(action_seq)
        entropy = dist.entropy()
        return log_probs, entropy, values


# ============================================================
# Rollout buffer
# ============================================================

@dataclass
class RolloutBatch:
    obs_seq: torch.Tensor
    actions_seq: torch.Tensor
    old_logp_seq: torch.Tensor
    returns_seq: torch.Tensor
    adv_seq: torch.Tensor
    hidden_init: torch.Tensor
    action_mask_seq: torch.Tensor


class RecurrentRolloutBuffer:
    def __init__(
        self,
        rollout_length: int,
        obs_dim: int,
        hidden_dim: int,
        n_actions: int,
        device: torch.device,
    ) -> None:
        self.rollout_length = rollout_length
        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim
        self.n_actions = n_actions
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.hidden_states: List[np.ndarray] = []
        self.action_masks: List[np.ndarray] = []

        self.returns: Optional[np.ndarray] = None
        self.advantages: Optional[np.ndarray] = None

    def add(
        self,
        obs: np.ndarray,
        action: int,
        log_prob: float,
        value: float,
        reward: float,
        done: bool,
        hidden_state: torch.Tensor,
        action_mask: np.ndarray,
    ) -> None:
        self.obs.append(np.asarray(obs, dtype=np.float32))
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.values.append(float(value))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.hidden_states.append(hidden_state.detach().cpu().numpy().astype(np.float32))
        self.action_masks.append(np.asarray(action_mask, dtype=np.float32))

    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> None:
        n = len(self.rewards)
        values = np.asarray(self.values + [last_value], dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)

        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0

        for t in reversed(range(n)):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[t] = gae

        self.advantages = advantages
        self.returns = advantages + values[:-1]

    def iter_sequence_batches(
        self,
        sequence_chunk_length: int,
        shuffle: bool,
        normalize_advantages: bool,
    ) -> Iterator[RolloutBatch]:
        if self.advantages is None or self.returns is None:
            raise RuntimeError("Call compute_gae() first")

        n = len(self.obs)
        starts = list(range(0, n, sequence_chunk_length))
        if shuffle:
            random.shuffle(starts)

        advantages = self.advantages.copy()
        if normalize_advantages:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for start in starts:
            end = min(start + sequence_chunk_length, n)

            obs_seq = np.stack(self.obs[start:end], axis=0)
            actions_seq = np.asarray(self.actions[start:end], dtype=np.int64)
            old_logp_seq = np.asarray(self.log_probs[start:end], dtype=np.float32)
            returns_seq = np.asarray(self.returns[start:end], dtype=np.float32)
            adv_seq = np.asarray(advantages[start:end], dtype=np.float32)
            hidden_init = self.hidden_states[start]
            action_mask_seq = np.stack(self.action_masks[start:end], axis=0)

            yield RolloutBatch(
                obs_seq=to_tensor(obs_seq[:, None, :], self.device),
                actions_seq=to_tensor(actions_seq[:, None], self.device, dtype=torch.long),
                old_logp_seq=to_tensor(old_logp_seq[:, None], self.device),
                returns_seq=to_tensor(returns_seq[:, None], self.device),
                adv_seq=to_tensor(adv_seq[:, None], self.device),
                hidden_init=to_tensor(hidden_init, self.device),
                action_mask_seq=to_tensor(action_mask_seq[:, None, :], self.device),
            )


# ============================================================
# PPO config
# ============================================================

@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    learning_rate: float = 1e-5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    rollout_length: int = 128
    sequence_chunk_length: int = 16
    normalize_advantages: bool = True


# ============================================================
# Checkpoint helpers
# ============================================================

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rollout_idx: int,
    global_step: int,
    best_val_ratio: float,
    config: PPOConfig,
    metadata: Dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rollout_idx": rollout_idx,
        "global_step": global_step,
        "best_val_ratio": best_val_ratio,
        "ppo_config": asdict(config),
        "metadata": metadata,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    return {
        "rollout_idx": int(payload.get("rollout_idx", 0)),
        "global_step": int(payload.get("global_step", 0)),
        "best_val_ratio": float(payload.get("best_val_ratio", -float("inf"))),
        "metadata": payload.get("metadata", {}),
    }


# ============================================================
# Trainer
# ============================================================

class RecurrentPPOTrainer:
    def __init__(
        self,
        train_env: RouteDatasetRiskEnv,
        val_env: RouteDatasetRiskEnv,
        test_env: RouteDatasetRiskEnv,
        model: RecurrentPPONet,
        config: PPOConfig,
        device: torch.device,
        checkpoint_dir: Path,
        run_metadata: Dict[str, Any],
        logger: logging.Logger,
    ) -> None:
        self.train_env = train_env
        self.val_env = val_env
        self.test_env = test_env
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.run_metadata = run_metadata
        self.logger = logger

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.buffer = RecurrentRolloutBuffer(
            rollout_length=config.rollout_length,
            obs_dim=train_env.obs_dim,
            hidden_dim=model.gru_hidden_dim,
            n_actions=train_env.action_space.n_actions,
            device=device,
        )

        self.obs, info = self.train_env.reset()
        self.current_action_mask = np.asarray(info["action_mask"], dtype=np.float32)
        self.hidden = self.model.initial_hidden(batch_size=1, device=device)

        self.global_step = 0
        self.episode_return = 0.0
        self.episode_length = 0
        self.episode_count = 0
        self.best_val_ratio = -float("inf")
        self.start_rollout_idx = 1

    def resume_from_checkpoint(self, ckpt_path: Path) -> None:
        info = load_checkpoint(
            path=ckpt_path,
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
        )

        self.global_step = info["global_step"]
        self.best_val_ratio = info["best_val_ratio"]
        self.start_rollout_idx = info["rollout_idx"] + 1

        self.logger.info(
            f"Resumed from checkpoint: {ckpt_path} | "
            f"start_rollout_idx={self.start_rollout_idx} | "
            f"global_step={self.global_step} | "
            f"best_val_ratio={self.best_val_ratio:.6f}"
        )

    @torch.no_grad()
    def collect_rollout(self) -> Dict[str, Any]:
        self.buffer.reset()

        finished_returns: List[float] = []
        finished_lengths: List[float] = []
        step_rewards: List[float] = []

        single_count = 0
        pair_count = 0
        z_before_list: List[float] = []
        z_after_list: List[float] = []
        restricted_count = 0

        total_steps = 0
        total_hits = 0
        chosen_rates: List[float] = []
        global_best_rates: List[float] = []

        for _ in range(self.config.rollout_length):
            obs_tensor = to_tensor(self.obs[None, :], self.device)
            mask_tensor = to_tensor(self.current_action_mask[None, :], self.device)
            hidden_at_action = self.hidden.clone()

            action_t, logp_t, value_t, next_hidden = self.model.act(
                obs=obs_tensor,
                h=self.hidden,
                action_mask=mask_tensor,
            )
            action = int(action_t.item())
            log_prob = float(logp_t.item())
            value = float(value_t.item())

            next_obs, reward, done, truncated, info = self.train_env.step(action)
            terminal = done or truncated

            step_rewards.append(float(reward))

            subset = info["subset"]
            if len(subset) == 1:
                single_count += 1
            elif len(subset) == 2:
                pair_count += 1

            z_before = float(info["z_before"])
            z_after = float(info["z_after"])
            z_before_list.append(z_before)
            z_after_list.append(z_after)

            if z_before >= 1.0 + self.train_env.epsilon:
                restricted_count += 1

            total_steps += 1
            if info.get("hit_global_best", False):
                total_hits += 1
            chosen_rates.append(float(info.get("chosen_rate", 0.0)))
            global_best_rates.append(float(info.get("global_best_rate", 0.0)))

            self.buffer.add(
                obs=self.obs,
                action=action,
                log_prob=log_prob,
                value=value,
                reward=reward,
                done=terminal,
                hidden_state=hidden_at_action,
                action_mask=self.current_action_mask,
            )

            self.obs = next_obs
            self.current_action_mask = np.asarray(info["action_mask"], dtype=np.float32)
            self.hidden = next_hidden
            self.global_step += 1
            self.episode_return += reward
            self.episode_length += 1

            if terminal:
                finished_returns.append(self.episode_return)
                finished_lengths.append(self.episode_length)

                self.episode_count += 1
                self.episode_return = 0.0
                self.episode_length = 0

                self.obs, reset_info = self.train_env.reset()
                self.current_action_mask = np.asarray(reset_info["action_mask"], dtype=np.float32)
                self.hidden = self.model.initial_hidden(batch_size=1, device=self.device)

        obs_tensor = to_tensor(self.obs[None, :], self.device)
        obs_seq = obs_tensor.unsqueeze(0)
        _, values, _ = self.model.forward_sequence(obs_seq, self.hidden)
        last_value = float(values[0, 0].item())

        self.buffer.compute_gae(
            last_value=last_value,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        total_action_count = single_count + pair_count

        mean_episode_return = float(np.mean(finished_returns)) if finished_returns else float("nan")
        mean_episode_length = float(np.mean(finished_lengths)) if finished_lengths else float("nan")

        per_episode_step_returns = [
            ret / max(length, 1.0)
            for ret, length in zip(finished_returns, finished_lengths)
        ]
        mean_step_return = (
            float(np.mean(per_episode_step_returns))
            if per_episode_step_returns else float("nan")
        )
        std_step_return = (
            float(np.std(per_episode_step_returns))
            if per_episode_step_returns else float("nan")
        )

        if not per_episode_step_returns and step_rewards:
            mean_step_return = float(np.mean(step_rewards))
            std_step_return = float(np.std(step_rewards))

        chosen_rates_arr = np.asarray(chosen_rates, dtype=np.float64)
        global_best_rates_arr = np.asarray(global_best_rates, dtype=np.float64)
        ratio = chosen_rates_arr / np.maximum(global_best_rates_arr, 1e-12)

        return {
            "finished_episodes": float(len(finished_returns)),
            "mean_episode_return": mean_episode_return,
            "mean_episode_length": mean_episode_length,
            "mean_step_return": mean_step_return,
            "std_step_return": std_step_return,
            "hit_global_best_rate": float(total_hits / total_steps) if total_steps > 0 else float("nan"),
            "chosen_over_best_ratio": float(np.mean(ratio)) if ratio.size > 0 else float("nan"),
            "single_action_ratio": float(single_count / total_action_count) if total_action_count > 0 else float("nan"),
            "pair_action_ratio": float(pair_count / total_action_count) if total_action_count > 0 else float("nan"),
            "mean_z_before": float(np.mean(z_before_list)) if z_before_list else float("nan"),
            "mean_z_after": float(np.mean(z_after_list)) if z_after_list else float("nan"),
            "max_z_before": float(np.max(z_before_list)) if z_before_list else float("nan"),
            "restricted_ratio": float(restricted_count / len(z_before_list)) if z_before_list else float("nan"),
            "z_trace": z_before_list,
        }
        
        
        
    def update(self) -> Dict[str, float]:
        policy_losses: List[float] = []
        value_losses: List[float] = []
        entropy_vals: List[float] = []
        total_losses: List[float] = []

        for _ in range(self.config.ppo_epochs):
            for batch in self.buffer.iter_sequence_batches(
                sequence_chunk_length=self.config.sequence_chunk_length,
                shuffle=True,
                normalize_advantages=self.config.normalize_advantages,
            ):
                log_probs, entropy, values = self.model.evaluate_actions_sequence(
                    obs_seq=batch.obs_seq,
                    h0=batch.hidden_init,
                    action_seq=batch.actions_seq,
                    action_mask_seq=batch.action_mask_seq,
                )

                ratio = torch.exp(log_probs - batch.old_logp_seq)
                surr1 = ratio * batch.adv_seq
                surr2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_eps,
                    1.0 + self.config.clip_eps,
                ) * batch.adv_seq

                policy_loss = -torch.min(surr1, surr2).mean()
                # value_loss = F.mse_loss(values, batch.returns_seq)
                
                value_scale = 200.0

                scaled_values = values / value_scale
                scaled_returns = batch.returns_seq / value_scale
                value_loss = F.mse_loss(scaled_values, scaled_returns)
                
                entropy_bonus = entropy.mean()

                total_loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    - self.config.entropy_coef * entropy_bonus
                )

                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_vals.append(float(entropy_bonus.item()))
                total_losses.append(float(total_loss.item()))

        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "entropy": float(np.mean(entropy_vals)),
            "total_loss": float(np.mean(total_losses)),
        }

    @torch.no_grad()
    def evaluate(
        self,
        env: RouteDatasetRiskEnv,
        n_episodes: int,
        show_progress: bool = False,
        desc: str = "Evaluating",
    ) -> Dict[str, float]:
        episode_returns: List[float] = []
        episode_lengths: List[int] = []
        episode_step_returns: List[float] = []

        total_steps = 0
        total_hits = 0
        chosen_rates: List[float] = []
        global_best_rates: List[float] = []
        z_values: List[float] = []

        single_count = 0
        pair_count = 0

        iterator = range(n_episodes)
        if show_progress:
            iterator = tqdm(iterator, desc=desc, leave=False)

        for _ in iterator:
            obs, info = env.reset()
            action_mask = np.asarray(info["action_mask"], dtype=np.float32)
            hidden = self.model.initial_hidden(batch_size=1, device=self.device)
            done = False
            truncated = False
            ep_ret = 0.0
            ep_len = 0

            while not (done or truncated):
                obs_tensor = to_tensor(obs[None, :], self.device)
                obs_seq = obs_tensor.unsqueeze(0)

                logits, _, next_hidden = self.model.forward_sequence(obs_seq, hidden)
                logits = logits[0, 0]

                mask_tensor = to_tensor(action_mask, self.device)
                masked_logits = self.model.apply_action_mask(logits, mask_tensor)
                action = int(torch.argmax(masked_logits, dim=-1).item())

                obs, reward, done, truncated, info = env.step(action)
                action_mask = np.asarray(info["action_mask"], dtype=np.float32)
                hidden = next_hidden if not (done or truncated) else self.model.initial_hidden(1, self.device)

                subset = info["subset"]
                if len(subset) == 1:
                    single_count += 1
                elif len(subset) == 2:
                    pair_count += 1

                ep_ret += reward
                ep_len += 1

                total_steps += 1
                total_hits += int(info.get("hit_global_best", False))
                chosen_rates.append(float(info.get("chosen_rate", 0.0)))
                global_best_rates.append(float(info.get("global_best_rate", 0.0)))
                z_values.append(float(info.get("z_before", 1.0)))

            episode_returns.append(ep_ret)
            episode_lengths.append(ep_len)
            episode_step_returns.append(ep_ret / max(ep_len, 1))

        chosen_rates_arr = np.asarray(chosen_rates, dtype=np.float64)
        global_best_rates_arr = np.asarray(global_best_rates, dtype=np.float64)
        ratio = chosen_rates_arr / np.maximum(global_best_rates_arr, 1e-12)

        total_action_count = single_count + pair_count

        return {
            "n_episodes": float(n_episodes),
            "mean_episode_return": float(np.mean(episode_returns)) if episode_returns else float("nan"),
            "std_episode_return": float(np.std(episode_returns)) if episode_returns else float("nan"),
            "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else float("nan"),
            "mean_step_return": float(np.mean(episode_step_returns)) if episode_step_returns else float("nan"),
            "std_step_return": float(np.std(episode_step_returns)) if episode_step_returns else float("nan"),
            "hit_global_best_rate": float(total_hits / total_steps) if total_steps > 0 else float("nan"),
            "mean_chosen_rate": float(np.mean(chosen_rates_arr)) if chosen_rates else float("nan"),
            "mean_global_best_rate": float(np.mean(global_best_rates_arr)) if global_best_rates else float("nan"),
            "chosen_over_best_ratio": float(np.mean(ratio)) if ratio.size > 0 else float("nan"),
            "single_action_ratio": float(single_count / total_action_count) if total_action_count > 0 else float("nan"),
            "pair_action_ratio": float(pair_count / total_action_count) if total_action_count > 0 else float("nan"),
            "mean_z": float(np.mean(z_values)) if z_values else float("nan"),
            "max_z": float(np.max(z_values)) if z_values else float("nan"),
        }

    def train(
        self,
        total_rollouts: int,
        validate_every: int,
        n_val_episodes: int,
        log_every: int,
    ) -> None:
        progress = tqdm(range(self.start_rollout_idx, total_rollouts + 1), desc="Training")

        for rollout_idx in progress:
            rollout_stats = self.collect_rollout()
            update_stats = self.update()

            if rollout_idx % log_every == 0:
                msg = (
                    f"[Rollout {rollout_idx}/{total_rollouts}] "
                    f"step={self.global_step} | "
                    f"ratio={rollout_stats['chosen_over_best_ratio']:.4f} | "
                    f"best_hit_rate={rollout_stats['hit_global_best_rate']:.4f} | "
                    f"avg_ret={rollout_stats['mean_step_return']:.4f} | "
                    f"single={rollout_stats['single_action_ratio']:.3f} | "
                    f"pair={rollout_stats['pair_action_ratio']:.3f} | "
                    f"z_mean={rollout_stats['mean_z_before']:.3f} | "
                    f"restricted={rollout_stats['restricted_ratio']:.3f} | "
                    f"pi_loss={update_stats['policy_loss']:.4f} | "
                    f"v_loss={update_stats['value_loss']:.4f} |"
                    f"entropy={update_stats['entropy']:.4f} | "
                    f"total_loss={update_stats['total_loss']:.4f}"
                )
                self.logger.info(msg)

            save_checkpoint(
                path=self.checkpoint_dir / "latest.pt",
                model=self.model,
                optimizer=self.optimizer,
                rollout_idx=rollout_idx,
                global_step=self.global_step,
                best_val_ratio=self.best_val_ratio,
                config=self.config,
                metadata=self.run_metadata,
            )

            if rollout_idx % validate_every == 0:
                if len(rollout_stats["z_trace"]) > 0:
                    save_z_plot(
                        z_values=rollout_stats["z_trace"],
                        epsilon=self.train_env.epsilon,
                        save_path=self.checkpoint_dir / "z_plots" / f"z_rollout_{rollout_idx}.png",
                        title=f"z over rollout {rollout_idx}",
                    )

                val_stats = self.evaluate(
                    self.val_env,
                    n_episodes=n_val_episodes,
                    show_progress=True,
                    desc=f"Validation {rollout_idx}",
                )
                self.logger.info(
                    f"[Validation @ rollout {rollout_idx}] "
                    f"avg_ret={val_stats['mean_step_return']:.4f} | "
                    f"ret_std={val_stats['std_step_return']:.4f} | "
                    f"ep_ret={val_stats['mean_episode_return']:.2f} | "
                    f"hit_best={val_stats['hit_global_best_rate']:.4f} | "
                    f"chosen/best={val_stats['chosen_over_best_ratio']:.4f} | "
                    f"pair={val_stats['pair_action_ratio']:.4f} | "
                    f"mean_z={val_stats['mean_z']:.4f}"
                )

                if val_stats["chosen_over_best_ratio"] > self.best_val_ratio:
                    self.best_val_ratio = float(val_stats["chosen_over_best_ratio"])
                    save_checkpoint(
                        path=self.checkpoint_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        rollout_idx=rollout_idx,
                        global_step=self.global_step,
                        best_val_ratio=self.best_val_ratio,
                        config=self.config,
                        metadata=self.run_metadata,
                    )
                    self.logger.info(
                        f"\nNew best checkpoint saved: best.pt | "
                        f"best_val_ratio={self.best_val_ratio:.6f}\n"
                    )


# ============================================================
# Main
# ============================================================

def main() -> None:
    # --------------------------------------------------------
    # Edit config here
    # --------------------------------------------------------
    seed = 42
    dataset_dir = "../dataset/CQI_arr8_medium_R_2_region_3"
    checkpoint_root = Path("./checkpoints_recurrent_ppo_risk")

    # split
    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1

    # fixed route length
    route_len = 1200

    # train / eval episode setup
    train_episode_horizon = 512
    eval_episode_horizon = route_len   # full-route for val/test
    train_random_start = True
    cache_routes = True

    # risk control
    alpha = 0.05
    epsilon = 0.2

    # model
    encoder_hidden_dim = 256
    gru_hidden_dim = 256

    # training
    total_rollouts = 3000
    validate_every = 50
    n_val_episodes = 20
    n_test_episodes = 50
    log_every = 20

    # resume
    resume_training = False
    resume_checkpoint_path = checkpoint_root / "run_xxx" / "latest.pt"

    # PPO
    ppo_config = PPOConfig(
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        entropy_coef=0.02,
        value_coef=0.05,
        learning_rate=5e-5,
        max_grad_norm=0.5,
        ppo_epochs=4,
        rollout_length=512,
        sequence_chunk_length=64,
        normalize_advantages=True,
    )

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = list_route_files(dataset_dir)
    split_dict = split_route_files(
        files=files,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    sample_T, n_cell, n_arr = inspect_dataset_shape(split_dict["train"])
    if sample_T != route_len:
        raise ValueError(
            f"Configured route_len={route_len}, but dataset sample length is {sample_T}. "
            f"Please set route_len correctly."
        )

    action_space = SubsetActionSpace(n_arr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = checkpoint_root / f"run_arr8_Ratio_alpha_{alpha}_eps_{epsilon}_horizon_{train_episode_horizon}_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(checkpoint_dir)
    logger.info(f"Using device: {device}")
    logger.info(f"Output dir: {checkpoint_dir}")

    log_config_block(logger, "data", {
        "dataset_dir": dataset_dir,
        "n_files_total": len(files),
        "train_files": len(split_dict["train"]),
        "val_files": len(split_dict["val"]),
        "test_files": len(split_dict["test"]),
        "route_len": route_len,
        "n_cell": n_cell,
        "n_arr": n_arr,
    })
    log_config_block(logger, "episode", {
        "train_episode_horizon": train_episode_horizon,
        "eval_episode_horizon": eval_episode_horizon,
        "train_random_start": train_random_start,
        "cache_routes": cache_routes,
        "rollout_length": ppo_config.rollout_length,
    })
    log_config_block(logger, "risk", {
        "alpha": alpha,
        "epsilon": epsilon,
        "threshold_b": 1.0 + epsilon,
    })
    log_config_block(logger, "model", {
        "encoder_hidden_dim": encoder_hidden_dim,
        "gru_hidden_dim": gru_hidden_dim,
        "n_actions": action_space.n_actions,
        "action_list": action_space.action_list,
    })
    log_config_block(logger, "training", {
        "total_rollouts": total_rollouts,
        "validate_every": validate_every,
        "n_val_episodes": n_val_episodes,
        "n_test_episodes": n_test_episodes,
        "log_every": log_every,
        "resume_training": resume_training,
        "resume_checkpoint_path": str(resume_checkpoint_path),
    })
    log_config_block(logger, "ppo", asdict(ppo_config))

    # --------------------------------------------------------
    # Envs
    # --------------------------------------------------------
    train_env = RouteDatasetRiskEnv(
        route_files=split_dict["train"],
        alpha=alpha,
        epsilon=epsilon,
        episode_horizon=train_episode_horizon,
        seed=seed,
        training=True,
        random_start=train_random_start,
        cache_routes=cache_routes,
    )

    val_env = RouteDatasetRiskEnv(
        route_files=split_dict["val"],
        alpha=alpha,
        epsilon=epsilon,
        episode_horizon=eval_episode_horizon,
        seed=seed + 1,
        training=False,
        random_start=False,
        cache_routes=cache_routes,
    )

    test_env = RouteDatasetRiskEnv(
        route_files=split_dict["test"],
        alpha=alpha,
        epsilon=epsilon,
        episode_horizon=eval_episode_horizon,
        seed=seed + 2,
        training=False,
        random_start=False,
        cache_routes=cache_routes,
    )

    model = RecurrentPPONet(
        obs_dim=train_env.obs_dim,
        n_actions=train_env.action_space.n_actions,
        encoder_hidden_dim=encoder_hidden_dim,
        gru_hidden_dim=gru_hidden_dim,
    )

    save_json(
        {
            "timestamp": timestamp,
            "output_dir": str(checkpoint_dir),
            "seed": seed,
            "dataset_dir": str(dataset_dir),
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "route_len": route_len,
            "train_episode_horizon": train_episode_horizon,
            "eval_episode_horizon": eval_episode_horizon,
            "train_random_start": train_random_start,
            "cache_routes": cache_routes,
            "alpha": alpha,
            "epsilon": epsilon,
            "threshold_b": 1.0 + epsilon,
            "n_cell": n_cell,
            "n_arr": n_arr,
            "encoder_hidden_dim": encoder_hidden_dim,
            "gru_hidden_dim": gru_hidden_dim,
            "action_list": [list(x) for x in action_space.action_list],
            "split_files": {
                "train": [str(p) for p in split_dict["train"]],
                "val": [str(p) for p in split_dict["val"]],
                "test": [str(p) for p in split_dict["test"]],
            },
            "training_config": {
                "total_rollouts": total_rollouts,
                "validate_every": validate_every,
                "n_val_episodes": n_val_episodes,
                "n_test_episodes": n_test_episodes,
                "log_every": log_every,
                "resume_training": resume_training,
                "resume_checkpoint_path": str(resume_checkpoint_path),
            },
            "ppo_config": asdict(ppo_config),
        },
        checkpoint_dir / "run_config.json",
    )

    trainer = RecurrentPPOTrainer(
        train_env=train_env,
        val_env=val_env,
        test_env=test_env,
        model=model,
        config=ppo_config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        run_metadata={
            "timestamp": timestamp,
            "n_cell": n_cell,
            "n_arr": n_arr,
            "route_len": route_len,
            "alpha": alpha,
            "epsilon": epsilon,
            "threshold_b": 1.0 + epsilon,
            "dataset_dir": str(dataset_dir),
            "encoder_hidden_dim": encoder_hidden_dim,
            "gru_hidden_dim": gru_hidden_dim,
            "ppo_config": asdict(ppo_config),
        },
        logger=logger,
    )

    if resume_training:
        trainer.resume_from_checkpoint(resume_checkpoint_path)

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    trainer.train(
        total_rollouts=total_rollouts,
        validate_every=validate_every,
        n_val_episodes=n_val_episodes,
        log_every=log_every,
    )

    # --------------------------------------------------------
    # Test with best checkpoint
    # --------------------------------------------------------
    best_ckpt = checkpoint_dir / "best.pt"
    if best_ckpt.exists():
        load_checkpoint(best_ckpt, trainer.model, optimizer=None, device=device)
        logger.info(f"Loaded best checkpoint for testing: {best_ckpt}")
    else:
        logger.warning("best.pt not found, testing current model instead")

    test_stats = trainer.evaluate(
        test_env,
        n_episodes=n_test_episodes,
        show_progress=True,
        desc="Testing",
    )
    logger.info("=== Full-Route Test Results ===")
    for k, v in test_stats.items():
        logger.info(f"{k:<24}: {v}")

    save_json(test_stats, checkpoint_dir / "test_metrics.json")


if __name__ == "__main__":
    main()