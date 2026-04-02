#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone runner with:
- parameters edited inside this file (no CLI args)
- auto experiment folder naming
- logging to both stdout and experiment log file
- only saves:
    1) building_mask_preview.pdf
    2) building_mask.npz
    3) capacity_map_yaw*_pitch*.npz
    4) CapacityMap_...yaw*_pitch*.pdf (vector)

Run:
    python Capacity_map_TX2.py

tmux example:
    tmux new -s caprun
    conda activate sionna-gpu
    python -u Capacity_map_TX2.py |& tee -a logs/caprun.log
"""

# ============================================================
# CONFIG (EDIT HERE ONLY)
# ============================================================

from pathlib import Path
import os

THIS_DIR = Path(__file__).resolve().parent
os.chdir(THIS_DIR)

CONFIG = {
    # ---------- TF / GPU ----------
    # None = keep existing env
    # ""   = force CPU
    # "0"  = GPU 0
    "CUDA_VISIBLE_DEVICES": 1,

    # ---------- scene / config ----------
    "scene_name": "nyu_tandon",
    "config_yaml": "config_height.yaml",

    # ---------- grid ----------
    "center": (-300.0, 200.0, 80.0),
    "grid_size": 5.0,
    "grid_dim": 200,

    # ---------- UE orientations (yaw, pitch) in degrees ----------
    "ue_orientations": [
        (0.0, 0.0),
    ],

    # ---------- plotting ----------
    "max_labels": 32,
    "alpha_mask": 0.15,
    "figsize": (8, 7),

    # ---------- output root ----------
    "experiments_root": "experiments_CM_height",
    "append_timestamp": True,

    # ---------- mask preview ----------
    "save_mask_preview": True,
}

# ============================================================
# Environment setup (set CUDA_VISIBLE_DEVICES BEFORE TF import)
# ============================================================

if CONFIG["CUDA_VISIBLE_DEVICES"] is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(CONFIG["CUDA_VISIBLE_DEVICES"])
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ============================================================
# Imports
# ============================================================

import re
import time
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

# headless matplotlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns
from matplotlib.colors import ListedColormap, BoundaryNorm

import yaml
from tqdm import tqdm

import tensorflow as tf

import mitsuba as mi
mi.set_log_level(mi.LogLevel.Error)
import drjit as dr

from sionna.rt import load_scene, ITURadioMaterial

import sionnautils
from sionnautils.custom_scene import list_scenes, get_scene

from Engine_V3 import Engine

# ============================================================
# Logging
# ============================================================

def setup_logger(exp_dir: Path):
    log_file = exp_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # clear existing handlers to avoid duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)

    return log_file

# ============================================================
# Helpers
# ============================================================

def print_gpu_info():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logging.info(f"GPU available: {gpus}")
    else:
        logging.info("No GPU, using CPU")
    logging.info(f"CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES')}")


def normalize_config(obj):
    """Recursively normalize numeric strings -> float; np.* constants -> values."""
    if isinstance(obj, dict):
        return {k: normalize_config(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [normalize_config(v) for v in obj]
    if isinstance(obj, str):
        s = obj.strip()
        if s == "np.pi":
            return np.pi
        if s == "np.e":
            return np.e
        if s in ("np.inf", "inf"):
            return np.inf
        if s in ("-np.inf", "-inf"):
            return -np.inf
        try:
            return float(s)
        except ValueError:
            return obj
    return obj


def sanitize_filename(s: str) -> str:
    s = str(s).strip().replace(" ", "_")
    s = re.sub(r"[^\w\-_.]", "", s)
    return s[:200] if len(s) > 200 else s


def short_center(center):
    return f"c{center[0]:g}_{center[1]:g}_{center[2]:g}"


def make_experiment_dir():
    root = Path(CONFIG["experiments_root"])
    root.mkdir(parents=True, exist_ok=True)

    scene = sanitize_filename(CONFIG["scene_name"])
    gdim = CONFIG["grid_dim"]
    gsz = CONFIG["grid_size"]
    cstr = short_center(CONFIG["center"])
    nori = len(CONFIG["ue_orientations"])

    name = f"{scene}_g{gdim}_s{gsz:g}_{cstr}_oris{nori}"
    if CONFIG.get("append_timestamp", True):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"{name}_{ts}"

    exp_dir = root / name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def generate_index_and_coordinate_map(center, grid_size, grid_dim):
    half_extent = grid_size * grid_dim / 2
    coords = np.zeros((grid_dim, grid_dim, 3), dtype=np.float32)
    for i in range(grid_dim):
        for j in range(grid_dim):
            x = center[0] - half_extent + (j + 0.5) * grid_size
            y = center[1] + half_extent - (i + 0.5) * grid_size
            z = center[2]
            coords[i, j] = (x, y, z)
    return coords


def build_xy_coords(center, grid_size, grid_dim):
    half_extent = (grid_dim * grid_size) / 2
    x_center, y_center = center[0], center[1]

    x_coords = np.linspace(
        x_center - half_extent + grid_size / 2,
        x_center + half_extent - grid_size / 2,
        grid_dim
    )

    y_coords = np.linspace(
        y_center + half_extent - grid_size / 2,
        y_center - half_extent + grid_size / 2,
        grid_dim
    )
    return x_coords, y_coords


def is_inside_building_mitsuba(scene, point, direction=np.array([0, 0, 1]), max_hits=50):
    """Ray casting intersection parity test."""
    direction = np.array([0.37, 0.23, 0.90], dtype=float)
    direction /= np.linalg.norm(direction)
    
    ray_origin = mi.Point3f(point)
    ray_dir = mi.Vector3f(direction)
    ray = mi.Ray3f(o=ray_origin, d=ray_dir)

    scene_mi = scene._scene
    count = 0

    for _ in range(max_hits):
        si = scene_mi.ray_intersect(ray, active=dr.ones(mi.Bool, 1)[0])
        if not si.is_valid():
            break
        count += 1
        ray.o = si.p + 1e-4 * ray.d

    return (count % 2 == 1)


def build_building_mask_and_preview(scene, coords, exp_dir: Path, grid_dim: int):
    """
    Produces:
    - building_mask in memory
    - building_mask.npz
    - building_mask_preview.pdf
    """
    logging.info("[Mask] Generating building mask (no caching) ...")
    t0 = time.time()

    building_mask = np.zeros((grid_dim, grid_dim), dtype=np.float32)

    pbar = tqdm(total=grid_dim * grid_dim, desc="Building mask", unit="pt")
    for i in range(grid_dim):
        for j in range(grid_dim):
            if is_inside_building_mitsuba(scene, coords[i, j]):
                building_mask[i, j] = 1.0
            pbar.update(1)
    pbar.close()

    mask_path = exp_dir / "building_mask.npz"
    np.savez_compressed(mask_path, building_mask=building_mask)
    logging.info(f"[Saved] {mask_path}")

    if CONFIG["save_mask_preview"]:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.contour(np.flipud(building_mask), levels=[0.5], colors="black", linewidths=1.2)
        ax.set_title("Building mask preview")
        out_path = exp_dir / "building_mask_preview.pdf"
        plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        logging.info(f"[Saved] {out_path}")

    logging.info(f"[TIME] building_mask took {time.time() - t0:.2f}s")
    return building_mask


def plot_best_txrx_selection(
    capacity_map,
    building_mask,
    x_coords,
    y_coords,
    tx_loc,
    cfg,
    ue_id=0,
    title="Capacity Map: best (TX,RX) selection",
    figsize=(8, 7),
    alpha_mask=0.15,
    max_labels=32,
    exp_dir: Path | None = None,
):
    """
    Saves vector PDF to exp_dir; returns saved path.
    """
    x_dim, y_dim, n_tx, n_rx = capacity_map.shape

    cap_flat = capacity_map.reshape(x_dim, y_dim, n_tx * n_rx)
    best_flat = np.argmax(cap_flat, axis=2)
    best_C = np.max(cap_flat, axis=2)

    best_index = best_flat.astype(int)
    best_index[building_mask == 1] = -1
    best_index[best_C == 0] = -1

    K = n_tx * n_rx
    base_colors = [
        "white",
        "#4C72B0", "#55A868", "#DD8452", "#C44E52",
        "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
        "#64B5CD", "#E17C05", "#76B7B2", "#F28E2B",
    ]
    if K + 1 > len(base_colors):
        raise ValueError(f"Need more colors: K={K} but only {len(base_colors)-1} provided")

    cmap = ListedColormap(base_colors[:K + 1])
    boundaries = [-1.5, -0.5] + [i + 0.5 for i in range(K)]
    norm = BoundaryNorm(boundaries, cmap.N)

    fig, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        best_index,
        cmap=cmap,
        norm=norm,
        square=True,
        linewidths=0,
        cbar=True,
        cbar_kws={"shrink": 0.75},
        ax=ax
    )

    ny, nx = best_index.shape
    step_x = max(1, nx // max_labels)
    step_y = max(1, ny // max_labels)
    xticks = np.arange(0, nx, step_x)
    yticks = np.arange(0, ny, step_y)

    ax.set_xticks(xticks + 0.5)
    ax.set_yticks(yticks + 0.5)
    ax.set_xticklabels(
        np.linspace(x_coords[0], x_coords[-1], len(xticks)).astype(int),
        rotation=90,
        ha="right"
    )
    ax.set_yticklabels(
        np.linspace(y_coords[0], y_coords[-1], len(yticks)).astype(int),
        rotation=0,
        ha="right"
    )

    ax.imshow(
        building_mask,
        origin="upper",
        extent=(0, nx, ny, 0),
        cmap=ListedColormap(["none", "black"]),
        alpha=alpha_mask,
        interpolation="nearest",
        zorder=5
    )
    ax.contour(
        building_mask.astype(float),
        levels=[0.5],
        origin="upper",
        extent=(0, nx, ny, 0),
        colors="black",
        linewidths=1.5,
        zorder=6
    )

    for tx_id, (x0, y0, z0) in enumerate(tx_loc):
        ix = int(np.argmin(np.abs(x_coords - x0)))
        iy = int(np.argmin(np.abs(y_coords - y0)))
        ax.scatter(ix, iy, s=220, c="red", marker="x", linewidths=3, zorder=50)

    cbar = ax.collections[0].colorbar
    ticks = list(range(K))
    cbar.set_ticks(ticks)

    labels = []
    for k in ticks:
        tx_id = k // n_rx
        rx_id = k % n_rx
        fc_ghz = cfg["ue"]["rx_fc"][rx_id] / 1e9
        labels.append(f"tx-{tx_id} | ue-{ue_id} | {fc_ghz:.1f}GHz | rx-{rx_id}")
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=8)
    cbar.ax.invert_yaxis()

    ax.set_title(title)
    plt.tight_layout()

    if exp_dir is None:
        raise ValueError("exp_dir is required for saving")

    fname = sanitize_filename(title)
    out_path = exp_dir / f"{fname}.pdf"
    plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    logging.info(f"[Saved] {out_path}")
    return out_path

# ============================================================
# Main
# ============================================================

def main():
    exp_dir = make_experiment_dir()
    log_file = setup_logger(exp_dir)

    logging.info(f"[CWD] Changed working directory to {THIS_DIR}")
    logging.info(f"[Experiment dir] {exp_dir}")
    logging.info(f"[Log file] {log_file}")
    logging.info(f"[Script] Start")

    total_t0 = time.time()

    print_gpu_info()

    with open(CONFIG["config_yaml"], "r") as f:
        cfg = normalize_config(yaml.safe_load(f))
    logging.info(f"[Config] Loaded {CONFIG['config_yaml']}")

    scene_path, map_data = get_scene(CONFIG["scene_name"])
    logging.info(f"[Scene] {CONFIG['scene_name']} -> {scene_path}")
    for k, v in map_data.items():
        logging.info(f"[SceneMeta] {k}: {v}")

    scene = load_scene(scene_path, merge_shapes=True)

    floor = scene.get("ground")
    floor.radio_material = ITURadioMaterial(
        "itu_concrete",
        "concrete",
        thickness=0.01,
        color=(0.5, 0.5, 0.5)
    )

    scene.remove("itu_wet_ground")

    for name, obj in scene.objects.items():
        try:
            logging.info(f"[Material] {name:<15}{obj.radio_material.name}")
        except Exception:
            logging.info(f"[Material] {name:<15}unknown")

    engine = Engine(cfg, scene)

    center = CONFIG["center"]
    grid_size = CONFIG["grid_size"]
    grid_dim = CONFIG["grid_dim"]

    coords = generate_index_and_coordinate_map(
        center=center,
        grid_size=grid_size,
        grid_dim=grid_dim
    )
    x_coords, y_coords = build_xy_coords(center, grid_size, grid_dim)

    building_mask = build_building_mask_and_preview(scene, coords, exp_dir, grid_dim)

    for yaw, pitch in CONFIG["ue_orientations"]:
        ori_t0 = time.time()
        logging.info(f"=== Run UE orientation yaw={yaw}, pitch={pitch} ===")

        engine.set_ue_orientation([[float(yaw), float(pitch)]])
        capacity_map = engine.run_capacity_map(coords)

        cap_path = exp_dir / f"capacity_map_z_{center[2]:g}_yaw{yaw:g}_pitch{pitch:g}.npz"
        np.savez_compressed(cap_path, capacity_map=capacity_map)
        logging.info(f"[Saved] {cap_path}")

        title = (
            f"CapacityMap g{grid_dim}_s{grid_size:g} center{center} "
            f"best (TX,RX)_UE rotation {engine.ue_list[0].body.orientation}"
        )

        plot_best_txrx_selection(
            capacity_map=capacity_map,
            building_mask=building_mask,
            x_coords=x_coords,
            y_coords=y_coords,
            tx_loc=engine.tx_loc,
            cfg=cfg,
            ue_id=0,
            title=title,
            figsize=CONFIG["figsize"],
            alpha_mask=CONFIG["alpha_mask"],
            max_labels=CONFIG["max_labels"],
            exp_dir=exp_dir,
        )

        logging.info(f"[TIME] yaw={yaw}, pitch={pitch} took {time.time() - ori_t0:.2f}s")

    logging.info(f"[DONE] All orientations finished.")
    logging.info(f"[TIME] total took {time.time() - total_t0:.2f}s")


if __name__ == "__main__":
    main()