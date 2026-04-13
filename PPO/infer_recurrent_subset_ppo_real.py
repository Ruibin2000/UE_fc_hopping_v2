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

    logger = logging.getLogger("ppo_inference")
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
# Environment
# ============================================================

class RouteDatasetSubsetEnv:
    def __init__(
        self,
        route_files: List[Path],
        lambda_e: float,
        episode_horizon: int,
        seed: int,
        training: bool,
        random_start: bool = False,
        cache_routes: bool = True,
    ) -> None:
        if len(route_files) == 0:
            raise ValueError("route_files must not be empty")

        self.route_files = list(route_files)
        self.lambda_e = float(lambda_e)
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

    @property
    def obs_dim(self) -> int:
        channels = self.n_cell * self.n_arr
        return channels * 3 + self.action_space.n_actions + 1

    def _load_route(self, path: Path) -> np.ndarray:
        if self.cache_routes:
            return self.route_cache[str(path)]
        return load_route_capacity(path)

    def _build_obs(self) -> np.ndarray:
        staleness = np.zeros((self.n_cell, self.n_arr), dtype=np.float32)
        never_seen = self.last_observed_time < 0
        staleness[never_seen] = float(max(self.episode_horizon, self.steps_in_episode + 1))
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
            ],
            axis=0,
        ).astype(np.float32)

        return obs

    def reset_with_route(self, route_path: Path, start_t: int = 0) -> Tuple[np.ndarray, Dict[str, Any]]:
        route = self._load_route(route_path)

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

        obs = self._build_obs()
        info = {
            "route_file": route_path.name,
            "route_length": int(route.shape[0]),
            "start_t": int(start_t),
        }
        return obs, info

    def step(
        self,
        action_idx: int,
        stop_at_horizon: bool = True,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_route is None:
            raise RuntimeError("Call reset() before step()")

        subset = self.action_space.idx_to_subset(action_idx)
        rates = self.current_route[self.current_t]  # [N_cell, N_arr]
        observed = np.full_like(rates, fill_value=np.nan)

        for arr_idx in subset:
            observed[:, arr_idx] = rates[:, arr_idx]

        observed_mask = np.isfinite(observed)

        self.last_observed_mask = observed_mask.astype(np.float32)
        self.last_observed_rate = np.where(observed_mask, observed, self.last_observed_rate).astype(np.float32)

        for c in range(self.n_cell):
            for a in range(self.n_arr):
                if observed_mask[c, a]:
                    self.last_observed_time[c, a] = self.steps_in_episode

        if np.isfinite(observed).any():
            chosen_rate = float(np.nanmax(observed))
            serving_flat_idx = int(np.nanargmax(observed))
            serving_cell, serving_arr = np.unravel_index(serving_flat_idx, observed.shape)
            serving_channel = (int(serving_cell), int(serving_arr))
        else:
            chosen_rate = 0.0
            serving_channel = None

        penalty = self.lambda_e if len(subset) == 2 else 0.0
        reward = chosen_rate - penalty

        global_best_rate = float(np.max(rates))
        global_best_flat_idx = int(np.argmax(rates))
        global_best_cell, global_best_arr = np.unravel_index(global_best_flat_idx, rates.shape)
        global_best_channel = (int(global_best_cell), int(global_best_arr))

        hit_global_best = bool(np.isclose(chosen_rate, global_best_rate, rtol=1e-6, atol=1e-8))

        self.prev_reward = float(reward)
        self.prev_action_idx = int(action_idx)

        self.current_t += 1
        self.steps_in_episode += 1

        route_done = self.current_t >= self.current_route.shape[0]
        horizon_done = self.steps_in_episode >= self.episode_horizon if stop_at_horizon else False
        done = bool(route_done or horizon_done)
        truncated = False

        next_obs = self._build_obs()
        info = {
            "route_file": self.current_route_path.name if self.current_route_path else None,
            "subset": subset,
            "action_idx": int(action_idx),
            "chosen_rate": chosen_rate,
            "penalty": penalty,
            "reward": reward,
            "serving_channel": serving_channel,
            "global_best_rate": global_best_rate,
            "global_best_channel": global_best_channel,
            "hit_global_best": hit_global_best,
            "time_index": int(self.current_t - 1),
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


# ============================================================
# Config / checkpoint
# ============================================================

@dataclass
class RunConfig:
    seed: int
    lambda_E: float
    episode_horizon: int
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
        raise KeyError(
            "run_config.json does not contain 'split_files'. "
            "Please re-run training after saving split_files."
        )

    return RunConfig(
        seed=int(cfg["seed"]),
        lambda_E=float(cfg["lambda_E"]),
        episode_horizon=int(cfg["episode_horizon"]),
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
    env: RouteDatasetSubsetEnv,
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
    penalty_list = []
    serving_channel_list = []
    global_best_channel_list = []

    single_count = 0
    pair_count = 0

    while not (done or truncated):
        obs_tensor = to_tensor(obs[None, :], device)
        obs_seq = obs_tensor.unsqueeze(0)

        logits, _, next_hidden = model.forward_sequence(obs_seq, hidden)
        logits = logits[0, 0]
        action = int(torch.argmax(logits, dim=-1).item())

        obs, reward, done, truncated, info = env.step(action, stop_at_horizon=False)
        hidden = next_hidden
        episode_return += reward

        frame_idx_list.append(int(info["time_index"]))
        action_idx_list.append(int(info["action_idx"]))
        subset_list.append(list(info["subset"]))
        chosen_rate_list.append(float(info["chosen_rate"]))
        global_best_rate_list.append(float(info["global_best_rate"]))
        hit_global_best_list.append(bool(info["hit_global_best"]))
        reward_list.append(float(info["reward"]))
        penalty_list.append(float(info["penalty"]))

        subset_len = len(info["subset"])
        if subset_len == 1:
            single_count += 1
        elif subset_len == 2:
            pair_count += 1

        if info["serving_channel"] is None:
            serving_channel_list.append([-1, -1])
        else:
            serving_channel_list.append(list(info["serving_channel"]))

        global_best_channel_list.append(list(info["global_best_channel"]))

    chosen_arr = np.asarray(chosen_rate_list, dtype=np.float32)
    best_arr = np.asarray(global_best_rate_list, dtype=np.float32)
    hit_arr = np.asarray(hit_global_best_list, dtype=np.bool_)
    reward_arr = np.asarray(reward_list, dtype=np.float32)

    mean_chosen_rate = float(np.mean(chosen_arr)) if chosen_arr.size > 0 else float("nan")
    mean_best_rate = float(np.mean(best_arr)) if best_arr.size > 0 else float("nan")
    mean_reward_per_step = float(np.mean(reward_arr)) if reward_arr.size > 0 else float("nan")
    hit_rate = float(np.mean(hit_arr)) if hit_arr.size > 0 else float("nan")
    ratio = float(mean_chosen_rate / mean_best_rate) if mean_best_rate > 0 else float("nan")

    total_action_count = single_count + pair_count
    single_action_ratio = (
        float(single_count / total_action_count) if total_action_count > 0 else float("nan")
    )
    pair_action_ratio = (
        float(pair_count / total_action_count) if total_action_count > 0 else float("nan")
    )

    route_result = {
        "route_file": route_path.name,
        "route_path": str(route_path),
        "route_length": int(reset_info["route_length"]),
        "episode_return": float(episode_return),
        "mean_reward_per_step": mean_reward_per_step,
        "hit_global_best_rate": hit_rate,
        "mean_chosen_rate": mean_chosen_rate,
        "mean_global_best_rate": mean_best_rate,
        "chosen_over_best_ratio": ratio,
        "single_action_count": int(single_count),
        "pair_action_count": int(pair_count),
        "single_action_ratio": single_action_ratio,
        "pair_action_ratio": pair_action_ratio,
        "frame_idx": np.asarray(frame_idx_list, dtype=np.int32),
        "action_idx": np.asarray(action_idx_list, dtype=np.int32),
        "subset": subset_list,
        "chosen_rate": chosen_arr,
        "global_best_rate": best_arr,
        "hit_global_best": hit_arr,
        "reward": reward_arr,
        "penalty": np.asarray(penalty_list, dtype=np.float32),
        "serving_channel": np.asarray(serving_channel_list, dtype=np.int32),
        "global_best_channel": np.asarray(global_best_channel_list, dtype=np.int32),
    }

    return route_result


def save_route_npz(route_result: Dict[str, Any], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    subset_raw = route_result["subset"]
    T = len(subset_raw)

    subset_mask = np.zeros((T, 2), dtype=np.bool_)
    subset_padded = np.full((T, 2), fill_value=-1, dtype=np.int32)

    for i, subset in enumerate(subset_raw):
        subset = list(subset)
        subset_padded[i, :len(subset)] = subset
        subset_mask[i, :len(subset)] = True

    np.savez_compressed(
        save_path,
        route_file=np.array(route_result["route_file"]),
        route_path=np.array(route_result["route_path"]),
        route_length=np.int32(route_result["route_length"]),
        episode_return=np.float32(route_result["episode_return"]),
        mean_reward_per_step=np.float32(route_result["mean_reward_per_step"]),
        hit_global_best_rate=np.float32(route_result["hit_global_best_rate"]),
        mean_chosen_rate=np.float32(route_result["mean_chosen_rate"]),
        mean_global_best_rate=np.float32(route_result["mean_global_best_rate"]),
        chosen_over_best_ratio=np.float32(route_result["chosen_over_best_ratio"]),
        single_action_count=np.int32(route_result["single_action_count"]),
        pair_action_count=np.int32(route_result["pair_action_count"]),
        single_action_ratio=np.float32(route_result["single_action_ratio"]),
        pair_action_ratio=np.float32(route_result["pair_action_ratio"]),
        frame_idx=route_result["frame_idx"],
        action_idx=route_result["action_idx"],
        subset_padded=subset_padded,
        subset_mask=subset_mask,
        chosen_rate=route_result["chosen_rate"],
        global_best_rate=route_result["global_best_rate"],
        hit_global_best=route_result["hit_global_best"],
        reward=route_result["reward"],
        penalty=route_result["penalty"],
        serving_channel=route_result["serving_channel"],
        global_best_channel=route_result["global_best_channel"],
    )


@torch.no_grad()
def infer_route_list_full(
    model: RecurrentPPONet,
    env: RouteDatasetSubsetEnv,
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
    total_single_count = 0
    total_pair_count = 0

    for route_idx, route_path in enumerate(route_files):
        route_result = infer_one_full_route(
            model=model,
            env=env,
            route_path=route_path,
            device=device,
        )

        out_name = f"{route_idx:04d}_{sanitize_stem(route_path)}.npz"
        save_route_npz(route_result, route_npz_dir / out_name)

        per_route_summary.append(
            {
                "route_idx": route_idx,
                "route_file": route_result["route_file"],
                "route_path": route_result["route_path"],
                "route_length": route_result["route_length"],
                "episode_return": route_result["episode_return"],
                "mean_reward_per_step": route_result["mean_reward_per_step"],
                "hit_global_best_rate": route_result["hit_global_best_rate"],
                "mean_chosen_rate": route_result["mean_chosen_rate"],
                "mean_global_best_rate": route_result["mean_global_best_rate"],
                "chosen_over_best_ratio": route_result["chosen_over_best_ratio"],
                "single_action_count": route_result["single_action_count"],
                "pair_action_count": route_result["pair_action_count"],
                "single_action_ratio": route_result["single_action_ratio"],
                "pair_action_ratio": route_result["pair_action_ratio"],
                "npz_file": out_name,
            }
        )

        episode_returns.append(route_result["episode_return"])
        mean_rewards.append(route_result["mean_reward_per_step"])
        hit_rates.append(route_result["hit_global_best_rate"])
        ratios.append(route_result["chosen_over_best_ratio"])
        single_ratios.append(route_result["single_action_ratio"])
        pair_ratios.append(route_result["pair_action_ratio"])
        total_single_count += int(route_result["single_action_count"])
        total_pair_count += int(route_result["pair_action_count"])

        logger.info(
            f"[{route_idx + 1:4d}/{len(route_files):4d}] "
            f"{route_path.name} | "
            f"ret={route_result['episode_return']:.4f} | "
            f"mean_reward={route_result['mean_reward_per_step']:.4f} | "
            f"hit={route_result['hit_global_best_rate']:.4f} | "
            f"ratio={route_result['chosen_over_best_ratio']:.4f} | "
            f"single_ratio={route_result['single_action_ratio']:.4f} | "
            f"pair_ratio={route_result['pair_action_ratio']:.4f}"
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
        "total_single_action_count": int(total_single_count),
        "total_pair_action_count": int(total_pair_count),
        "overall_single_action_ratio": (
            float(total_single_count / total_action_count) if total_action_count > 0 else float("nan")
        ),
        "overall_pair_action_ratio": (
            float(total_pair_count / total_action_count) if total_action_count > 0 else float("nan")
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
    run_dir = Path("./checkpoints_ppo/run_lambda_100.0_lr_1e-05_20260407_142342")
    use_best_checkpoint = True
    split_name = "test"   # "train" / "val" / "test"

    # output folder
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

    sample = load_route_capacity(route_files[0])
    n_cell = sample.shape[1]
    n_arr = sample.shape[2]
    action_space = SubsetActionSpace(n_arr)
    obs_dim = n_cell * n_arr * 3 + action_space.n_actions + 1

    env = RouteDatasetSubsetEnv(
        route_files=route_files,
        lambda_e=cfg.lambda_E,
        episode_horizon=cfg.episode_horizon,
        seed=cfg.seed + 999,
        training=False,
        random_start=False,
        cache_routes=True,
    )

    model = load_model_from_checkpoint(
        ckpt_path=ckpt_path,
        obs_dim=obs_dim,
        n_actions=action_space.n_actions,
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
    logger.info(f"{'torch_cuda_available':<28}: {torch.cuda.is_available()}")
    logger.info(f"{'torch_cuda_count':<28}: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        try:
            logger.info(f"{'current_cuda_device':<28}: {torch.cuda.current_device()}")
            logger.info(
                f"{'cuda_device_name':<28}: "
                f"{torch.cuda.get_device_name(torch.cuda.current_device())}"
            )
        except Exception as e:
            logger.info(f"{'cuda_device_info_error':<28}: {e}")

    log_config_block(logger, "RUN CONFIG", {
        "seed": cfg.seed,
        "lambda_E": cfg.lambda_E,
        "episode_horizon": cfg.episode_horizon,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "gru_hidden_dim": cfg.gru_hidden_dim,
        "num_routes_in_split": len(route_files),
        "split_name": split_name,
        "use_best_checkpoint": use_best_checkpoint,
    })

    log_config_block(logger, "DATA SHAPE", {
        "n_cell": n_cell,
        "n_arr": n_arr,
        "n_actions": action_space.n_actions,
        "obs_dim": obs_dim,
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
    logger.info(f"{'overall_single_action_ratio':<28}: {summary['overall_single_action_ratio']:.6f}")
    logger.info(f"{'overall_pair_action_ratio':<28}: {summary['overall_pair_action_ratio']:.6f}")
    logger.info(f"{'total_single_action_count':<28}: {summary['total_single_action_count']}")
    logger.info(f"{'total_pair_action_count':<28}: {summary['total_pair_action_count']}")
    logger.info(f"{'route_npz_dir':<28}: {route_npz_dir}")
    logger.info(f"{'summary_json':<28}: {summary_json}")


if __name__ == "__main__":
    main()