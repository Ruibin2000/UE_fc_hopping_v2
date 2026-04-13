#!/usr/bin/env python3
# route_gen_clean.py
#
# Logging version:
# - Fixes CUDA_VISIBLE_DEVICES issue (must set before importing TensorFlow)
# - Keeps Mitsuba inside-building test
# - Adds max_attempts_per_step to prevent infinite rejection loops
# - If all attempts fail at a step: TURN AROUND IN PLACE (yaw += pi)
# - Uses logging instead of print
# - Writes logs to ./logs/route_log_YYYYMMDD_HHMMSS.log

# =====================
# TensorFlow / GPU
# =====================
import os

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0  # 使用 CPU 可设为 ""
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

# ============================================================
# 1) Imports
# ============================================================
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
import gymnasium as gym

from tqdm import tqdm

import mitsuba as mi
import drjit as dr
from scipy.spatial.transform import Rotation as R
from matplotlib.path import Path as MplPath

from sionna.rt import load_scene, ITURadioMaterial

import sionnautils
from sionnautils.custom_scene import list_scenes, get_scene

mi.set_log_level(mi.LogLevel.Error)   # 只显示 Error，不显示 Warn


# ============================================================
# 2) Logger helpers
# ============================================================
def setup_logger():
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("route_gen")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 防止重复运行时重复添加 handler
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    # logger.addHandler(stream_handler)

    logger.info("Logger initialized")
    logger.info(f"Log file: {log_file}")

    return logger, log_file


def log_dict(logger, title, d):
    logger.info(title)
    for k, v in d.items():
        logger.info(f"{k:20s}: {v}")
    logger.info("=" * 40)


# ============================================================
# 3) Config helpers
# ============================================================
def normalize_config(obj):
    """Recursively normalize YAML:
    - numeric strings (incl scientific notation) -> float
    - 'np.pi', 'np.e', 'np.inf' -> numpy constants
    """
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


def load_cfg(cfg_path="config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return normalize_config(cfg)


# ============================================================
# 4) Geometry helpers
# ============================================================
def wrap_yaw(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def wrap_pitch(a):
    return (a + np.pi / 2) % np.pi - np.pi / 2


def is_point_in_region(region_vertices_3d, point_3d):
    """region: (N,3), point: (3,) -> bool in XY polygon."""
    region_vertices_3d = np.asarray(region_vertices_3d, dtype=np.float32)
    point_3d = np.asarray(point_3d, dtype=np.float32)
    polygon_xy = region_vertices_3d[:, :2]
    point_xy = point_3d[:2]
    return MplPath(polygon_xy).contains_point(point_xy)


# ============================================================
# 5) Mitsuba inside-building test
# ============================================================
def is_inside_building_mitsuba(scene, point, direction=np.array([0.37, 0.23, 0.90]), max_hits=50):
    """
    Ray parity test: odd #hits -> inside.
    direction uses a non-axis-aligned vector to reduce degeneracy.

    NOTE: Most reliable if meshes are watertight. If your scene isn't watertight,
    consider a multi-direction voting version later.
    """
    point = np.asarray(point, dtype=np.float32)
    direction = np.asarray(direction, dtype=np.float32)
    direction = direction / (np.linalg.norm(direction) + 1e-12)

    ray = mi.Ray3f(o=mi.Point3f(point), d=mi.Vector3f(direction))
    scene_mi = scene._scene  # Sionna wraps Mitsuba internally

    count = 0
    for _ in range(max_hits):
        si = scene_mi.ray_intersect(ray, active=True)
        if not si.is_valid():
            break
        count += 1
        ray.o = si.p + 1e-4 * ray.d  # avoid self-hit

    return (count % 2 == 1)


def ue_inside_building(scene, cfg, position, rotation, check_center_first=True):
    """
    position: (3,) or (n,3)
    rotation: (3,) or (n,3) radians, order [yaw, pitch, roll] for "zyx"
    cfg["ue"]["rx_loc_pos"]: (n_rx,3) UE-local RX offsets
    """
    rx_offset = np.asarray(cfg["ue"]["rx_loc_pos"], dtype=np.float32)

    position = np.atleast_2d(np.asarray(position, dtype=np.float32))
    rotation = np.atleast_2d(np.asarray(rotation, dtype=np.float32))

    n_ue = position.shape[0]
    for i in range(n_ue):
        if check_center_first and is_inside_building_mitsuba(scene, position[i]):
            return True

        r = R.from_euler("zyx", rotation[i], degrees=False)
        rx_positions = r.apply(rx_offset) + position[i]  # (n_rx,3)

        if any(is_inside_building_mitsuba(scene, p) for p in rx_positions):
            return True

    return False


# ============================================================
# 6) UE state init + route generation
# ============================================================
def make_action_space(cfg):
    rand_seed = int(cfg["motion"]["rand_seed"])
    dv_min, dv_max = cfg["motion"]["dv_min"], cfg["motion"]["dv_max"]
    dphi_min = np.deg2rad(cfg["motion"]["dphi_min"])
    dphi_max = np.deg2rad(cfg["motion"]["dphi_max"])
    dt = float(cfg["motion"]["measure_period"])

    action_space = gym.spaces.Box(
        low=np.array([dv_min * dt, dphi_min * dt], dtype=np.float64),
        high=np.array([dv_max * dt, dphi_max * dt], dtype=np.float64),
        dtype=np.float64,
    )
    action_space.seed(rand_seed)
    np.random.seed(rand_seed)
    return action_space


def initialize_ue_state(scene, cfg, region, z=1.0, max_tries=200):
    region = np.asarray(region, dtype=np.float32)

    for _ in range(max_tries):
        x = np.random.uniform(region[:, 0].min(), region[:, 0].max())
        y = np.random.uniform(region[:, 1].min(), region[:, 1].max())
        p0 = np.array([x, y, z], dtype=np.float32)

        if not is_point_in_region(region, p0):
            continue

        # walking yaw only
        yaw_walk0 = np.random.uniform(-np.pi, np.pi)
        walk_rot0 = np.array([yaw_walk0, 0.0, 0.0], dtype=np.float32)

        # spin yaw + pitch
        yaw_spin0 = np.random.uniform(-np.pi, np.pi)
        pitch_spin0 = np.random.uniform(-np.pi / 2, np.pi / 2)
        spin_rot0 = np.array([yaw_spin0, pitch_spin0, 0.0], dtype=np.float32)

        ue_rot0 = np.array(
            [wrap_yaw(walk_rot0[0] + spin_rot0[0]), spin_rot0[1], 0.0],
            dtype=np.float32,
        )

        if ue_inside_building(scene, cfg, p0, ue_rot0):
            continue

        return p0, walk_rot0, spin_rot0

    raise RuntimeError("initialize_ue_state failed: region too strict or inside-test too aggressive.")


def generate_routes(scene, cfg, region, action_space):
    """
    Returns:
      routes: list[n_ue] each is list[step] of [x,y,z]
      rotations: list[n_ue] each is list[step] of [yaw,pitch,roll]
      walk_rots: list[n_ue] each is list[step] of [yaw,0,0]
      spin_rots: list[n_ue] each is list[step] of [yaw,pitch,0]
    """
    n_ue = int(cfg["ue"]["n_ue"])
    dt = float(cfg["motion"]["measure_period"])
    measure_time = float(cfg["motion"]["measure_time"])
    n_steps = int(measure_time / dt)

    v_init = float(cfg["motion"]["v_intial"])
    v_max = float(cfg["motion"]["max_speed"])
    v_min = float(cfg["motion"]["min_speed"])
    R_speed_level = float(cfg["motion"]["R_speed_level"])

    max_attempts_per_step = int(cfg["motion"].get("max_attempts_per_step", 200))

    route_list = []
    rotation_list = []
    walk_rot_list = []
    spin_rot_list = []

    for _idx_ue in range(n_ue):
        p0, walk_rot, spin_rot = initialize_ue_state(scene, cfg, region, z=1.0)
        x, y, z = map(float, p0)
        v = v_init

        route = [[x, y, z]]
        ue_rot_seq = []
        walk_rot_seq = []
        spin_rot_seq = []

        ue_rot = np.array(
            [wrap_yaw(walk_rot[0] + spin_rot[0]), spin_rot[1], 0.0],
            dtype=np.float32,
        )
        ue_rot_seq.append(ue_rot.tolist())
        walk_rot_seq.append(walk_rot.tolist())
        spin_rot_seq.append(spin_rot.tolist())

        step = 0
        while step < n_steps - 1:
            accepted = False

            for _ in range(max_attempts_per_step):
                dv, dphi = action_space.sample()

                # 1) walking update
                v = float(np.clip(v + dv, v_min, v_max))
                walk_rot[0] = wrap_yaw(walk_rot[0] + dphi)

                # 2) spin update
                step_yaw = np.random.uniform(
                    -R_speed_level * np.pi * dt * 2,
                    R_speed_level * np.pi * dt * 2,
                )
                step_pitch = np.random.uniform(
                    -R_speed_level * np.pi * dt,
                    R_speed_level * np.pi * dt,
                )
                spin_rot[0] = wrap_yaw(spin_rot[0] + step_yaw)
                spin_rot[1] = wrap_pitch(spin_rot[1] + step_pitch)

                # 3) compose UE rotation
                ue_rot = np.array(
                    [wrap_yaw(walk_rot[0] + spin_rot[0]), wrap_pitch(spin_rot[1]), 0.0],
                    dtype=np.float32,
                )

                # 4) propose move by walking yaw
                temp_x = x + v * np.cos(walk_rot[0]) * dt
                temp_y = y + v * np.sin(walk_rot[0]) * dt
                test_point = np.array([temp_x, temp_y, z], dtype=np.float32)

                # reject if outside or inside building
                if (not is_point_in_region(region, test_point)) or ue_inside_building(scene, cfg, test_point, ue_rot):
                    walk_rot[0] = wrap_yaw(walk_rot[0] + np.random.choice([-1, 1]) * np.pi / 6)
                    continue

                # accept
                x, y = float(temp_x), float(temp_y)
                route.append([x, y, z])
                ue_rot_seq.append(ue_rot.tolist())
                walk_rot_seq.append(walk_rot.tolist())
                spin_rot_seq.append(spin_rot.tolist())
                step += 1
                accepted = True
                break

            if not accepted:
                # Too many rejects: turn around in place
                turned = False
                for _ in range(12):
                    walk_rot[0] = wrap_yaw(walk_rot[0] + np.pi)
                    ue_rot_now = np.array(
                        [
                            wrap_yaw(walk_rot[0] + spin_rot[0]),
                            wrap_pitch(spin_rot[1]),
                            0.0,
                        ],
                        dtype=np.float32,
                    )

                    cur_point = np.array([x, y, z], dtype=np.float32)
                    if (not is_point_in_region(region, cur_point)) or ue_inside_building(scene, cfg, cur_point, ue_rot_now):
                        jitter = float(cfg["motion"].get("turnaround_jitter_rad", 0.1))
                        walk_rot[0] = wrap_yaw(walk_rot[0] + np.random.uniform(-jitter, jitter))
                        continue

                    turned = True
                    break

                # if still invalid, just continue; next proposals will be checked anyway
                continue

        route_list.append(route)
        rotation_list.append(ue_rot_seq)
        walk_rot_list.append(walk_rot_seq)
        spin_rot_list.append(spin_rot_seq)

    return route_list, rotation_list, walk_rot_list, spin_rot_list


# ============================================================
# 7) Scene loader
# ============================================================
def load_and_prepare_scene(cfg, logger):
    map_data = None

    if cfg["scene"]["name"] != "free_space":
        # 如果你想完全跟配置走，可以改成:
        # scene_path, map_data = get_scene(cfg["scene"]["name"])
        scene_path, map_data = get_scene("nyu_tandon")

        logger.info("===== Scene map data =====")
        for k, v in map_data.items():
            logger.info(f"{k}: {v}")
        logger.info("=" * 40)

        scene = load_scene(scene_path, merge_shapes=True)

        floor = scene.get("ground")
        floor.radio_material = ITURadioMaterial(
            "itu_concrete", "concrete", thickness=0.01, color=(0.5, 0.5, 0.5)
        )

        try:
            scene.remove("itu_wet_ground")
            logger.info("Removed object: itu_wet_ground")
        except Exception:
            logger.warning("Object 'itu_wet_ground' not found, skip remove.")

        logger.info("===== Scene objects / materials =====")
        for name, obj in scene.objects.items():
            try:
                logger.info(f"{name:<20} {obj.radio_material.name}")
            except Exception:
                logger.info(f"{name:<20} <no radio material>")
        logger.info("=" * 40)
    else:
        scene = load_scene()
        logger.info("Loaded free_space scene")

    return scene, map_data


# ============================================================
# 8) Main save loop
# ============================================================
def main():
    logger, log_file = setup_logger()

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info("GPU available: %s", gpus)
    else:
        logger.info("No GPU, using CPU")
    logger.info("TF GPUs: %s", gpus)
    logger.info("Mitsuba variant: %s", mi.variant())
    logger.info("CUDA_VISIBLE_DEVICES = %s", os.getenv("CUDA_VISIBLE_DEVICES"))

    cfg = load_cfg("config_arr8_s3_medium.yaml")

    log_dict(logger, "===== Route parameters =====", cfg["route"])
    log_dict(logger, "===== Motion parameters =====", cfg["motion"])

    scene, map_data = load_and_prepare_scene(cfg, logger)

    if map_data is not None:
        logger.info("Scene map_data:")
        for k, v in map_data.items():
            logger.info(f"  {k}: {v}")

    region = np.array(cfg["motion"]["region"], dtype=np.float32)
    action_space = make_action_space(cfg)

    output_dir = Path(cfg["route"]["folder"])
    output_dir.mkdir(parents=True, exist_ok=True)

    n_routes = int(cfg["route"]["n_routes"])
    route_count = 0

    stats = {"skip": 0, "success": 0, "fail": 0}
    start_total = time.time()

    logger.info("Output dir: %s", output_dir)
    logger.info("n_routes: %d", n_routes)

    for i in tqdm(range(n_routes), desc="Generating routes"):
        file_path = output_dir / f"routes_{i:04d}.npz"

        if file_path.exists():
            stats["skip"] += 1
            logger.info(f"[SKIP] {file_path.name}")
            continue

        logger.info(f"[RUN ] {file_path.name}")
        t0 = time.time()

        try:
            r_list, rot_list, walk_list, spin_list = generate_routes(
                scene, cfg, region, action_space
            )

            positions = np.asarray(r_list, dtype=np.float32)
            rotations = np.asarray(rot_list, dtype=np.float32)
            walk_rots = np.asarray(walk_list, dtype=np.float32)
            spin_rots = np.asarray(spin_list, dtype=np.float32)

            np.savez(
                file_path,
                positions=positions,
                rotations=rotations,
                walk_rotations=walk_rots,
                spin_rotations=spin_rots,
                n_ue=positions.shape[0],
                n_step=positions.shape[1],
            )

            dt = time.time() - t0
            stats["success"] += 1
            route_count += 1

            logger.info(
                f"[OK  ] {file_path.name} | time={dt:.2f}s | "
                f"positions.shape={positions.shape} | rotations.shape={rotations.shape}"
            )

        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received. Exiting safely.")
            raise
        except Exception:
            stats["fail"] += 1
            logger.exception(f"[FAIL] {file_path.name}")
            continue

    total_time = time.time() - start_total

    logger.info("=" * 50)
    logger.info(f"Done. Output dir: {output_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Saved {route_count} new routes to {output_dir}")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60.0:.2f} min)")
    logger.info(f"Summary: {stats}")


if __name__ == "__main__":
    main()