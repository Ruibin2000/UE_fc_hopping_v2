#!/usr/bin/env python3
# route_gen_clean.py
#
# Clean version:
# - Fixes CUDA_VISIBLE_DEVICES issue (must set before importing TensorFlow)
# - Keeps your Mitsuba inside-building test
# - Adds max_attempts_per_step to prevent infinite rejection loops
# - If all attempts fail at a step: TURN AROUND IN PLACE (yaw += pi), no re-init, no step-forcing


# =====================
# TensorFlow / GPU
# =====================
import os
# import tensorflow as tf

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    gpu_num = 0  # 使用 CPU 可设为 ""
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPU available:", gpus)
else:
    print("No GPU, using CPU")

# if os.getenv("CUDA_VISIBLE_DEVICES") is None:
#     gpu_num = 0  # 使用 CPU 可设为 ""
#     os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ============================================================
# 1) Imports
# ============================================================
import json
from pathlib import Path

import numpy as np
import yaml
import gymnasium as gym

from tqdm import tqdm

import tensorflow as tf  # after env vars are set

import mitsuba as mi
import drjit as dr
from scipy.spatial.transform import Rotation as R
from matplotlib.path import Path as MplPath

from sionna.rt import load_scene, ITURadioMaterial

import sionnautils
from sionnautils.custom_scene import list_scenes, get_scene

mi.set_log_level(mi.LogLevel.Error)   # 只显示 Error，不显示 Warn

print("Mitsuba variant:", mi.variant())

# from Engine_V3 import Engine  # not used in this generator


# ============================================================
# 2) Config helpers
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
# 3) Geometry helpers
# ============================================================
def wrap_yaw(a):
    return (a + np.pi) % (2 * np.pi) - np.pi

def wrap_pitch(a):
    return (a + np.pi/2) % np.pi - np.pi/2

def is_point_in_region(region_vertices_3d, point_3d):
    """region: (N,3), point: (3,) -> bool in XY polygon."""
    region_vertices_3d = np.asarray(region_vertices_3d, dtype=np.float32)
    point_3d = np.asarray(point_3d, dtype=np.float32)
    polygon_xy = region_vertices_3d[:, :2]
    point_xy = point_3d[:2]
    return MplPath(polygon_xy).contains_point(point_xy)


# ============================================================
# 4) Mitsuba inside-building test
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
    rotation: (3,) or (n,3)  radians, order [yaw, pitch, roll] for "zyx"
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
# 5) UE state init + route generation
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
    R_speed_level = float(cfg["motion"]["R_speed_level"])

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

        ue_rot0 = np.array([walk_rot0[0] + spin_rot0[0],spin_rot0[1], 0.0], dtype=np.float32)

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

    # Key: avoid infinite while loops
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

        ue_rot = np.array([walk_rot[0] + spin_rot[0], spin_rot[1], 0.0], dtype=np.float32)
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
                step_yaw = np.random.uniform(-R_speed_level * np.pi * dt * 2,
                                             R_speed_level * np.pi * dt * 2)
                step_pitch = np.random.uniform(-R_speed_level * np.pi * dt,
                                               R_speed_level * np.pi * dt)
                spin_rot[0] = wrap_yaw(spin_rot[0] + step_yaw)
                spin_rot[1] = wrap_pitch(spin_rot[1] + step_pitch)

                # 3) compose UE rotation
                ue_rot = np.array([wrap_yaw(walk_rot[0] + spin_rot[0]), wrap_pitch(spin_rot[1]), 0.0], dtype=np.float32)

                # 4) propose move by walking yaw
                temp_x = x + v * np.cos(walk_rot[0]) * dt
                temp_y = y + v * np.sin(walk_rot[0]) * dt
                test_point = np.array([temp_x, temp_y, z], dtype=np.float32)

                # reject if outside or inside building
                if (not is_point_in_region(region, test_point)) or ue_inside_building(scene, cfg, test_point, ue_rot):
                    # mild nudge to escape local minima
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
                # Too many rejects: turn around in place (direction opposite)
                # Ensure CURRENT pose (x,y,z + new rotation) is also valid.
                turned = False
                for _ in range(12):  #最多尝试12次不同朝向
                    walk_rot[0] = wrap_yaw(walk_rot[0] + np.pi)
                    ue_rot_now = np.array([
                        wrap_yaw(walk_rot[0] + spin_rot[0]),
                        wrap_pitch(spin_rot[1]),
                        0.0
                    ], dtype=np.float32)

                    cur_point = np.array([x, y, z], dtype=np.float32)
                    if (not is_point_in_region(region, cur_point)) or ue_inside_building(scene, cfg, cur_point, ue_rot_now):
                        # still invalid, add a small random yaw tweak and retry
                        jitter = float(cfg["motion"].get("turnaround_jitter_rad", 0.1))
                        walk_rot[0] = wrap_yaw(walk_rot[0] + np.random.uniform(-jitter, jitter))
                        continue

                    turned = True
                    break

                # if still not turned to a valid pose, just keep going; next proposals will be checked anyway
                continue


        route_list.append(route)
        rotation_list.append(ue_rot_seq)
        walk_rot_list.append(walk_rot_seq)
        spin_rot_list.append(spin_rot_seq)

    return route_list, rotation_list, walk_rot_list, spin_rot_list


# ============================================================
# 6) Scene loader
# ============================================================
def load_and_prepare_scene(cfg):
    # =====================
    # Load scene
    # =====================
    if cfg["scene"]["name"] != "free_space":
        scene_path, map_data = get_scene('nyu_tandon')
        for k, v in map_data.items():
            print(f'{k}: {v}')

        scene = load_scene(scene_path,merge_shapes=True)

        floor = scene.get("ground")
        floor.radio_material = ITURadioMaterial(
            "itu_concrete", "concrete", thickness=0.01, color=(0.5, 0.5, 0.5)
        )

        scene.remove("itu_wet_ground")
        for name, obj in scene.objects.items():
            print(f'{name:<15}{obj.radio_material.name}')
    else:
        scene = load_scene()

    return scene, map_data


# ============================================================
# 7) Main save loop
# ============================================================
def main():
    print("TF GPUs:", tf.config.list_physical_devices("GPU"))

    cfg = load_cfg("config.yaml")

    print("===== Route parameters =====")
    for k, v in cfg["route"].items():
        print(f"{k:20s}: {v}")
    print("=============================\n")

    print("===== Motion parameters =====")
    for k, v in cfg["motion"].items():
        print(f"{k:20s}: {v}")
    print("=============================\n")


    # Scene name optional in YAML, default nyu_tandon
    # scene_name = cfg.get("scene", {}).get("name", "nyu_tandon")
    scene, map_data = load_and_prepare_scene(cfg)

    print("Scene map_data:")
    for k, v in map_data.items():
        print(f"  {k}: {v}")

    region = np.array(cfg["motion"]["region"], dtype=np.float32)
    action_space = make_action_space(cfg)

    output_dir = Path(cfg["route"]["folder"])
    output_dir.mkdir(parents=True, exist_ok=True)

    n_routes = int(cfg["route"]["n_routes"])
    route_count = 0

    for i in tqdm(range(n_routes), desc="Generating routes"):
        file_path = output_dir / f"routes_{i:04d}.npz"

        if file_path.exists():
            continue  # ⭐ 跳过已经生成的
        r_list, rot_list, walk_list, spin_list = generate_routes(scene, cfg, region, action_space)

        positions = np.asarray(r_list, dtype=np.float32)
        rotations = np.asarray(rot_list, dtype=np.float32)
        walk_rots = np.asarray(walk_list, dtype=np.float32)
        spin_rots = np.asarray(spin_list, dtype=np.float32)

        file_path = output_dir / f"routes_{i:04d}.npz"
        np.savez(
            file_path,
            positions=positions,
            rotations=rotations,
            walk_rotations=walk_rots,
            spin_rotations=spin_rots,
            n_ue=positions.shape[0],
            n_step=positions.shape[1],
        )


    print(f"✅ Saved {route_count} routes to {output_dir}")


if __name__ == "__main__":
    main()
