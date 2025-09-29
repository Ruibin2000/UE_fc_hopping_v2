import argparse
import logging
from pathlib import Path

import numpy as np
import torch
# import matplotlib.pyplot as plt
from torch.distributions import Bernoulli

from model_collections import best_obs_minus_wE_torch, epsilon_greedy_prob
from bandit_model import EpsilonBanditPolicy
from Env import Env
import json

from pathlib import Path

# ----------------- path -----------------
THIS_DIR = Path(__file__).resolve().parent

# ---------------------------
# Utility helpers
# ---------------------------

def index_to_list(index: int, length: int) -> list[int]:
    """Return a binary list with 1 at *index* and 0 elsewhere."""
    if index >= length:
        raise ValueError("Index exceeds the total number of antennas")
    return [1 if i == index else 0 for i in range(length)]

# Prepare binary combination utilities
def index_to_binary_list(index: int, length: int) -> list:
    if index >= 2 ** length:
        raise ValueError("Index exceeds the total number of combinations")
    return [int(b) for b in format(index, f'0{length}b')]

def total_combinations(length: int) -> int:
    return 2 ** length

# ---------------------------
# Testing + live logging
# ---------------------------

def test(env_path: str, rate_path: str, test_episodes: int, file_obs: str, file_mask:str):
    """Run ε-greedy evaluation and generate two diagnostic plots.

    1. best_score vs. ε-greedy reward (per step timeline)
    2. The four rate components (next_state[:4]) per step, along with
       max_rate_epsilon and max_rate_true (highlighted).
    """
    env = Env(env_path, rate_path)
    E = env.energy_consume_array  # Assumed constant over episode
    policy = EpsilonBanditPolicy(action_dim=total_combinations(4), epsilon=0.2, forbid_zero=True)

    # --- buffers for plotting ---
    t_all: list[int] = []
    best_scores: list[float] = []
    rewards: list[float] = []

    rate_components = [[], [], [], []]  # 4 separate lists
    max_rate_eps: list[float] = []
    max_rate_true: list[float] = []

    obs_history = []
    mask_history = []

    global_step = 0  # timeline across ALL episodes

    for epi in range(1, test_episodes + 1):
        _ = env.reset()
        # The env in this project seems to expose a helper to retrieve current states:
        state, next_state = env.step_true()  # ground-truth transition (no cost)
        done = False

        obs_history.append(state[:4].tolist())
        mask_history.append([1,1,1,1])

        while not done:
            # Greedy best (oracle) action for current *next_state*
            best_idx, best_score, best_obs = best_obs_minus_wE_torch(obs=next_state[:4], E=E, w=1.0)
            best_action = index_to_list(best_idx, 4)

            # ε-greedy probabilities derived from current *state*
            probs = policy.probs()
            action = policy.sample_action()

            # Step using ε-greedy action
            action_list = index_to_binary_list(action, 4)
            state, r, done, info = env.step(action_list)
            policy.update(action, r/(probs[action] + 1e-12)) 

            obs_history.append(state[:4].tolist())
            mask_history.append(action_list)

            # Compute diagnostic quantities
            max_rate_eps_step = env.max_rate(action_list)
            max_rate_true_step = env.max_rate(best_action)

            # --- log for plotting ---
            t_all.append(global_step)
            best_scores.append(best_score)
            rewards.append(r)

            # Store each of the 4 rate components from next_state (before step)
            for i in range(4):
                rate_components[i].append(next_state[i])
            max_rate_eps.append(max_rate_eps_step)
            max_rate_true.append(max_rate_true_step)

            # Advance timeline
            global_step += 1

            # Prepare for next loop iteration
            if not done:
                _, next_state = env.step_true()
            else:
                next_state = state  # final state

    with open(THIS_DIR / file_obs, 'w') as f:
        json.dump(obs_history, f)

    with open(THIS_DIR / file_mask, 'w') as f:
        json.dump(mask_history, f)
    

# ---------------------------
# CLI entry
# ---------------------------
if __name__ == "__main__":
    
    mask_folder = Path(THIS_DIR / "dataset_multi_5mps/newBandit_masked_dataset_0.2_150")
    mask_folder.mkdir(exist_ok=True, parents=True)


    for path_idx in range(41,61):
        for idx in range(0,50):
            
            file_obs = "dataset_multi_5mps/newBandit_masked_dataset_0.2_150/51_region_1_" + str(path_idx) + "_"+str(idx)+"_60s_R2_obs.json"
            file_mask = "dataset_multi_5mps/newBandit_masked_dataset_0.2_150/51_region_1_" + str(path_idx) + "_"+str(idx)+"_60s_R2_masks.json"
            file_snr = "dataset_multi_5mps/dataset_raw/51_region_1_" + str(path_idx) + "_"+str(idx)+"_60s_R2_snr.json"
            file_rate = "dataset_multi_5mps/dataset_raw/51_region_1_" + str(path_idx) + "_"+str(idx)+"_60s_R2_rate.json"

            test(THIS_DIR / file_snr, THIS_DIR / file_rate, 1, file_obs, file_mask)