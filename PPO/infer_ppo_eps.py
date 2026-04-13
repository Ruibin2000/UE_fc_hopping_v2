from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import json
import random
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
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


def sanitize_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def setup_logger(log_dir: Path, log_name: str = "inference.log") -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / log_name

    logger = logging.getLogger("ppo_inference_risk")
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


# ============================================================
# Action space
# ============================================================

class SubsetActionSpace:
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


# ============================================================
# Dataset
# ============================================================

def load_route_capacity(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)

    if "capacity" not in data:
        raise KeyError(f"'capacity' not found in {npz_path}")

    cap = np.squeeze(np.array(data["capacity"])).astype(float)

    if cap.ndim != 3:
        raise ValueError(f"Expected squeezed capacity to be 3D, got {cap.shape} in {npz_path}")

    # raw: [T, N_arr, N_cell] -> internal: [T, N_cell, N_arr]
    cap = cap.transpose(0, 2, 1)
    return cap.astype(np.float32)


# ============================================================
# Risk Environment (aligned with training)
# ============================================================

class RouteDatasetRiskEnv:
    def __init__(
        self,
        route_files: List[Path],
        alpha: float,
        epsilon: float,
        episode_horizon: int,
        seed: int,
        training: bool,
        random_start: bool = False,
        cache_routes: bool = True,
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

    def reset_with_route(self, route_path: Path, start_t: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        route = self._load_route(route_path)

        if start_t < 0 or start_t >= route.shape[0]:
            raise ValueError(f"Invalid start_t={start_t} for route length={route.shape[0]}")

        self.current_route = route
        self.current_route_path = route_path
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
            "route_file": route_path.name,
            "route_length": int(route.shape[0]),
            "start_t": int(start_t),
            "z": float(self.z),
            "action_mask": self._get_action_mask(),
        }
        return obs, info

    def step(
        self,
        action_idx: int,
        stop_at_horizon: bool = False,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_route is None:
            raise RuntimeError("Call reset_with_route() before step()")

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
        self.prev_action_idx = int(action_idx)

        self.current_t += 1
        self.steps_in_episode += 1

        route_done = self.current_t >= self.current_route.shape[0]
        horizon_done = self.steps_in_episode >= self.episode_horizon if stop_at_horizon else False
        done = bool(route_done or horizon_done)
        truncated = False

        next_obs = self._build_obs()
        next_mask = self._get_action_mask()

        info = {
            "route_file": self.current_route_path.name if self.current_route_path else None,
            "subset": subset,
            "action_idx": int(action_idx),
            "chosen_rate": chosen_rate,
            "reward": reward,
            "serving_channel": serving_channel,
            "global_best_rate": global_best_rate,
            "global_best_channel": global_best_channel,
            "hit_global_best": hit_global_best,
            "time_index": int(self.current_t - 1),
            "z_before": z_before,
            "z_after": z_after,
            "restricted_before_action": bool(z_before >= 1.0 + self.epsilon),
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
        obs_seq: torch.Tensor,   # [T, B, obs_dim]
        h0: torch.Tensor,        # [1, B, H]
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


# ============================================================
# Config / checkpoint
# ============================================================

@dataclass
class RunConfig:
    seed: int
    alpha: float
    epsilon: float
    train_episode_horizon: int
    eval_episode_horizon: int
    encoder_hidden_dim: int
    gru_hidden_dim: int
    split_files: Dict[str, List[str]]


def load_run_config(run_dir: Path) -> RunConfig:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing run_config.json: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    split_files = cfg.get("split_files", None)
    if split_files is None:
        raise KeyError("run_config.json does not contain 'split_files'")

    return RunConfig(
        seed=int(cfg["seed"]),
        alpha=float(cfg["alpha"]),
        epsilon=float(cfg["epsilon"]),
        train_episode_horizon=int(cfg["train_episode_horizon"]),
        eval_episode_horizon=int(cfg["eval_episode_horizon"]),
        encoder_hidden_dim=int(cfg["encoder_hidden_dim"]),
        gru_hidden_dim=int(cfg["gru_hidden_dim"]),
        split_files=split_files,
    )


def load_model_from_checkpoint(
    ckpt_path: Path,
    obs_dim: int,
    n_actions: int,
    encoder_hidden_dim: int,
    gru_hidden_dim: int,
    device: torch.device,
) -> RecurrentPPONet:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = RecurrentPPONet(
        obs_dim=obs_dim,
        n_actions=n_actions,
        encoder_hidden_dim=encoder_hidden_dim,
        gru_hidden_dim=gru_hidden_dim,
    ).to(device)

    payload = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


# ============================================================
# Inference
# ============================================================

@torch.no_grad()
def infer_one_full_route(
    model: RecurrentPPONet,
    env: RouteDatasetRiskEnv,
    route_path: Path,
    device: torch.device,
) -> Dict[str, Any]:
    obs, reset_info = env.reset_with_route(route_path, start_t=0)
    hidden = model.initial_hidden(batch_size=1, device=device)

    done = False
    truncated = False
    episode_return = 0.0

    frame_idx_list = []
    action_idx_list = []
    subset_list = []
    chosen_rate_list = []
    global_best_rate_list = []
    hit_global_best_list = []
    reward_list = []
    serving_channel_list = []
    global_best_channel_list = []
    z_before_list = []
    z_after_list = []
    restricted_before_list = []

    single_count = 0
    pair_count = 0
    restricted_count = 0

    while not (done or truncated):
        obs_tensor = to_tensor(obs[None, :], device)          # [1, obs_dim]
        obs_seq = obs_tensor.unsqueeze(0)                     # [1, 1, obs_dim]

        logits, _, next_hidden = model.forward_sequence(obs_seq, hidden)
        logits = logits[0, 0]

        action_mask_np = env._get_action_mask()
        action_mask = to_tensor(action_mask_np, device)
        masked_logits = model.apply_action_mask(logits, action_mask)

        action = int(torch.argmax(masked_logits, dim=-1).item())

        obs, reward, done, truncated, info = env.step(action, stop_at_horizon=False)
        hidden = next_hidden
        episode_return += reward

        subset = tuple(info["subset"])
        if len(subset) == 1:
            single_count += 1
        elif len(subset) == 2:
            pair_count += 1

        if info["restricted_before_action"]:
            restricted_count += 1

        frame_idx_list.append(int(info["time_index"]))
        action_idx_list.append(int(info["action_idx"]))
        subset_list.append(list(subset))
        chosen_rate_list.append(float(info["chosen_rate"]))
        global_best_rate_list.append(float(info["global_best_rate"]))
        hit_global_best_list.append(bool(info["hit_global_best"]))
        reward_list.append(float(info["reward"]))
        serving_channel_list.append(info["serving_channel"])
        global_best_channel_list.append(info["global_best_channel"])
        z_before_list.append(float(info["z_before"]))
        z_after_list.append(float(info["z_after"]))
        restricted_before_list.append(bool(info["restricted_before_action"]))

    chosen_arr = np.array(chosen_rate_list, dtype=np.float64)
    best_arr = np.array(global_best_rate_list, dtype=np.float64)

    ratio = np.divide(
        chosen_arr,
        best_arr,
        out=np.zeros_like(chosen_arr),
        where=best_arr > 0,
    )

    total_action_count = single_count + pair_count

    return {
        "route_file": route_path.name,
        "route_length": int(reset_info["route_length"]),
        "episode_return": float(episode_return),
        "num_steps": int(len(action_idx_list)),
        "mean_reward_per_step": float(np.mean(reward_list)) if reward_list else float("nan"),
        "hit_global_best_rate": float(np.mean(hit_global_best_list)) if hit_global_best_list else float("nan"),
        "chosen_over_best_ratio": float(np.mean(ratio)) if ratio.size > 0 else float("nan"),
        "single_action_count": int(single_count),
        "pair_action_count": int(pair_count),
        "single_action_ratio": float(single_count / total_action_count) if total_action_count > 0 else float("nan"),
        "pair_action_ratio": float(pair_count / total_action_count) if total_action_count > 0 else float("nan"),
        "restricted_count": int(restricted_count),
        "restricted_ratio": float(restricted_count / total_action_count) if total_action_count > 0 else float("nan"),
        "mean_z_before": float(np.mean(z_before_list)) if z_before_list else float("nan"),
        "mean_z_after": float(np.mean(z_after_list)) if z_after_list else float("nan"),
        "max_z_after": float(np.max(z_after_list)) if z_after_list else float("nan"),
        "trace": {
            "frame_idx": frame_idx_list,
            "action_idx": action_idx_list,
            "subset": subset_list,
            "chosen_rate": chosen_rate_list,
            "global_best_rate": global_best_rate_list,
            "hit_global_best": hit_global_best_list,
            "reward": reward_list,
            "serving_channel": serving_channel_list,
            "global_best_channel": global_best_channel_list,
            "z_before": z_before_list,
            "z_after": z_after_list,
            "restricted_before_action": restricted_before_list,
        },
    }


@torch.no_grad()
def infer_route_list_full(
    model: RecurrentPPONet,
    env: RouteDatasetRiskEnv,
    route_files: List[Path],
    device: torch.device,
    route_npz_dir: Path,
    logger: logging.Logger,
) -> Dict[str, Any]:
    per_route_summary: List[Dict[str, Any]] = []

    episode_returns = []
    mean_rewards = []
    hit_rates = []
    ratios = []
    single_ratios = []
    pair_ratios = []
    restricted_ratios = []
    mean_zs = []
    max_zs = []

    total_single_count = 0
    total_pair_count = 0
    total_restricted_count = 0

    iterator = tqdm(route_files, desc="Full-route inference")

    for route_idx, route_path in enumerate(iterator):
        route_result = infer_one_full_route(
            model=model,
            env=env,
            route_path=route_path,
            device=device,
        )

        out_name = route_npz_dir / f"{sanitize_stem(route_path)}_inference.npz"
        np.savez_compressed(
            out_name,
            **{
                "frame_idx": np.array(route_result["trace"]["frame_idx"], dtype=np.int32),
                "action_idx": np.array(route_result["trace"]["action_idx"], dtype=np.int32),
                "chosen_rate": np.array(route_result["trace"]["chosen_rate"], dtype=np.float32),
                "global_best_rate": np.array(route_result["trace"]["global_best_rate"], dtype=np.float32),
                "hit_global_best": np.array(route_result["trace"]["hit_global_best"], dtype=np.bool_),
                "reward": np.array(route_result["trace"]["reward"], dtype=np.float32),
                "z_before": np.array(route_result["trace"]["z_before"], dtype=np.float32),
                "z_after": np.array(route_result["trace"]["z_after"], dtype=np.float32),
                "restricted_before_action": np.array(route_result["trace"]["restricted_before_action"], dtype=np.bool_),
                "subset": np.array(route_result["trace"]["subset"], dtype=object),
                "serving_channel": np.array(route_result["trace"]["serving_channel"], dtype=object),
                "global_best_channel": np.array(route_result["trace"]["global_best_channel"], dtype=object),
            }
        )

        per_route_summary.append(
            {
                "route_file": route_result["route_file"],
                "route_length": route_result["route_length"],
                "episode_return": route_result["episode_return"],
                "num_steps": route_result["num_steps"],
                "mean_reward_per_step": route_result["mean_reward_per_step"],
                "hit_global_best_rate": route_result["hit_global_best_rate"],
                "chosen_over_best_ratio": route_result["chosen_over_best_ratio"],
                "single_action_ratio": route_result["single_action_ratio"],
                "pair_action_ratio": route_result["pair_action_ratio"],
                "restricted_ratio": route_result["restricted_ratio"],
                "mean_z_before": route_result["mean_z_before"],
                "mean_z_after": route_result["mean_z_after"],
                "max_z_after": route_result["max_z_after"],
                "npz_file": str(out_name),
            }
        )

        episode_returns.append(route_result["episode_return"])
        mean_rewards.append(route_result["mean_reward_per_step"])
        hit_rates.append(route_result["hit_global_best_rate"])
        ratios.append(route_result["chosen_over_best_ratio"])
        single_ratios.append(route_result["single_action_ratio"])
        pair_ratios.append(route_result["pair_action_ratio"])
        restricted_ratios.append(route_result["restricted_ratio"])
        mean_zs.append(route_result["mean_z_after"])
        max_zs.append(route_result["max_z_after"])

        total_single_count += int(route_result["single_action_count"])
        total_pair_count += int(route_result["pair_action_count"])
        total_restricted_count += int(route_result["restricted_count"])

        logger.info(
            f"[{route_idx + 1:4d}/{len(route_files):4d}] "
            f"{route_path.name} | "
            f"ratio={route_result['chosen_over_best_ratio']:.4f} | "
            f"best_hit_rate={route_result['hit_global_best_rate']:.4f} | "
            f"avg_ret={route_result['mean_reward_per_step']:.4f} | "
            f"single={route_result['single_action_ratio']:.3f} | "
            f"pair={route_result['pair_action_ratio']:.3f} | "
            f"z_mean={route_result['mean_z_after']:.3f} | "
            f"restricted={route_result['restricted_ratio']:.3f}"
        )

    total_action_count = total_single_count + total_pair_count

    return {
        "num_routes": len(route_files),
        "mean_return": float(np.mean(episode_returns)) if episode_returns else float("nan"),
        "mean_reward_per_step": float(np.mean(mean_rewards)) if mean_rewards else float("nan"),
        "mean_hit_global_best_rate": float(np.mean(hit_rates)) if hit_rates else float("nan"),
        "mean_chosen_over_best_ratio": float(np.mean(ratios)) if ratios else float("nan"),
        "mean_single_action_ratio": float(np.mean(single_ratios)) if single_ratios else float("nan"),
        "mean_pair_action_ratio": float(np.mean(pair_ratios)) if pair_ratios else float("nan"),
        "mean_restricted_ratio": float(np.mean(restricted_ratios)) if restricted_ratios else float("nan"),
        "mean_z": float(np.mean(mean_zs)) if mean_zs else float("nan"),
        "max_z": float(np.max(max_zs)) if max_zs else float("nan"),
        "total_single_action_count": int(total_single_count),
        "total_pair_action_count": int(total_pair_count),
        "overall_single_action_ratio": (
            float(total_single_count / total_action_count) if total_action_count > 0 else float("nan")
        ),
        "overall_pair_action_ratio": (
            float(total_pair_count / total_action_count) if total_action_count > 0 else float("nan")
        ),
        "overall_restricted_ratio": (
            float(total_restricted_count / total_action_count) if total_action_count > 0 else float("nan")
        ),
        "per_route": per_route_summary,
    }


# ============================================================
# Main
# ============================================================

def main() -> None:
    # --------------------------------------------------------
    # Edit config here
    # --------------------------------------------------------
    

    run_dir = Path(f"./checkpoints_recurrent_ppo_risk/run_arr8_Ratio_alpha_0.05_eps_0.2_horizon_1200_20260412_030537")
    use_best_checkpoint = True
    split_name = "test"   # "train" / "val" / "test"

    output_root = run_dir / f"inference_fullroute_{'best' if use_best_checkpoint else 'latest'}_{split_name}"
    route_npz_dir = output_root / "routes"
    summary_json = output_root / "summary.json"

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------
    ckpt_path = run_dir / ("best.pt" if use_best_checkpoint else "latest.pt")
    cfg = load_run_config(run_dir)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    route_files = [Path(p) for p in cfg.split_files[split_name]]
    if len(route_files) == 0:
        raise RuntimeError(f"No routes found in split: {split_name}")

    env = RouteDatasetRiskEnv(
        route_files=route_files,
        alpha=cfg.alpha,
        epsilon=cfg.epsilon,
        episode_horizon=cfg.eval_episode_horizon,
        seed=cfg.seed + 999,
        training=False,
        random_start=False,
        cache_routes=True,
    )

    model = load_model_from_checkpoint(
        ckpt_path=ckpt_path,
        obs_dim=env.obs_dim,
        n_actions=env.action_space.n_actions,
        encoder_hidden_dim=cfg.encoder_hidden_dim,
        gru_hidden_dim=cfg.gru_hidden_dim,
        device=device,
    )

    output_root.mkdir(parents=True, exist_ok=True)
    route_npz_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        output_root,
        log_name=f"inference_{'best' if use_best_checkpoint else 'latest'}_{split_name}.log",
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=== INFERENCE STARTED ===")
    logger.info(f"{'timestamp':<28}: {timestamp}")
    logger.info(f"{'run_dir':<28}: {run_dir}")
    logger.info(f"{'checkpoint':<28}: {ckpt_path}")
    logger.info(f"{'split':<28}: {split_name}")
    logger.info(f"{'output_root':<28}: {output_root}")
    logger.info(f"{'device':<28}: {device}")

    log_config_block(logger, "RUN CONFIG", {
        "seed": cfg.seed,
        "alpha": cfg.alpha,
        "epsilon": cfg.epsilon,
        "threshold_b": 1.0 + cfg.epsilon,
        "train_episode_horizon": cfg.train_episode_horizon,
        "eval_episode_horizon": cfg.eval_episode_horizon,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "gru_hidden_dim": cfg.gru_hidden_dim,
        "num_routes_in_split": len(route_files),
        "split_name": split_name,
        "use_best_checkpoint": use_best_checkpoint,
    })

    log_config_block(logger, "DATA SHAPE", {
        "n_cell": env.n_cell,
        "n_arr": env.n_arr,
        "n_actions": env.action_space.n_actions,
        "obs_dim": env.obs_dim,
    })

    summary = infer_route_list_full(
        model=model,
        env=env,
        route_files=route_files,
        device=device,
        route_npz_dir=route_npz_dir,
        logger=logger,
    )

    summary = {
        "run_dir": str(run_dir),
        "checkpoint": ckpt_path.name,
        "split": split_name,
        "output_root": str(output_root),
        **summary,
    }

    save_json(summary, summary_json)

    logger.info("=== FULL-ROUTE INFERENCE DONE ===")
    logger.info(f"{'run_dir':<28}: {run_dir}")
    logger.info(f"{'checkpoint':<28}: {ckpt_path.name}")
    logger.info(f"{'split':<28}: {split_name}")
    logger.info(f"{'num_routes':<28}: {summary['num_routes']}")
    logger.info(f"{'mean_return':<28}: {summary['mean_return']:.6f}")
    logger.info(f"{'mean_reward_per_step':<28}: {summary['mean_reward_per_step']:.6f}")
    logger.info(f"{'mean_hit_global_best_rate':<28}: {summary['mean_hit_global_best_rate']:.6f}")
    logger.info(f"{'mean_chosen_over_best_ratio':<28}: {summary['mean_chosen_over_best_ratio']:.6f}")
    logger.info(f"{'mean_single_action_ratio':<28}: {summary['mean_single_action_ratio']:.6f}")
    logger.info(f"{'mean_pair_action_ratio':<28}: {summary['mean_pair_action_ratio']:.6f}")
    logger.info(f"{'mean_restricted_ratio':<28}: {summary['mean_restricted_ratio']:.6f}")
    logger.info(f"{'mean_z':<28}: {summary['mean_z']:.6f}")
    logger.info(f"{'max_z':<28}: {summary['max_z']:.6f}")
    logger.info(f"{'overall_single_action_ratio':<28}: {summary['overall_single_action_ratio']:.6f}")
    logger.info(f"{'overall_pair_action_ratio':<28}: {summary['overall_pair_action_ratio']:.6f}")
    logger.info(f"{'overall_restricted_ratio':<28}: {summary['overall_restricted_ratio']:.6f}")
    logger.info(f"{'route_npz_dir':<28}: {route_npz_dir}")
    logger.info(f"{'summary_json':<28}: {summary_json}")


if __name__ == "__main__":
    main()