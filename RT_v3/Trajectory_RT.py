# =====================
# TensorFlow / GPU (MUST set env BEFORE importing TF)
# =====================
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# 只在没设置时才默认用 GPU 0；想强制 CPU：os.environ["CUDA_VISIBLE_DEVICES"] = ""
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("GPU available:" if gpus else "No GPU, using CPU", gpus)

# =====================
# Scientific stack
# =====================
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# =====================
# Mitsuba / DrJit
# =====================
import mitsuba as mi
import drjit as dr

# =====================
# Sionna RT
# =====================
from sionna.rt import (
    load_scene,
    PlanarArray,
    Transmitter,
    Receiver,
    Camera,
    PathSolver,
    ITURadioMaterial,
    SceneObject,
    AntennaPattern,
    register_antenna_pattern,
)

# =====================
# Custom utils / scenes
# =====================
import sionnautils
from sionnautils.custom_scene import list_scenes, get_scene

# =====================
# Project-specific
# =====================
from Engine_V3 import Engine

# =====================
# Misc
# =====================
from pathlib import Path
from tqdm import tqdm
import yaml
from collections import defaultdict

mi.set_log_level(mi.LogLevel.Error)   # 只显示 Error，不显示 Warn


def normalize_config(obj):
    """Recursively normalize:
    - numeric strings (incl. scientific notation) -> float
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


# =====================
# Load config
# =====================
with open("config.yaml", "r") as f:
    cfg = normalize_config(yaml.safe_load(f))


print("===== Route parameters =====")
for k, v in cfg["route"].items():
    print(f"{k:20s}: {v}")
print("=============================\n")

print("===== Motion parameters =====")
for k, v in cfg["motion"].items():
    print(f"{k:20s}: {v}")
print("=============================\n")

# =====================
# Load scene
# =====================
scenes = list_scenes()
print("Scenes:", scenes)

scene_file = scenes[0]
scene_path, map_data = get_scene(scene_file)

# for k, v in map_data.items():
#     print(f"{k}: {v}")

scene = load_scene(scene_path, merge_shapes=True)

floor = scene.get("ground")
floor.radio_material = ITURadioMaterial(
    "itu_concrete", "concrete", thickness=0.01, color=(0.5, 0.5, 0.5)
)

scene.remove("itu_wet_ground")

# for name, obj in scene.objects.items():
#     print(f"{name:<20}{obj.radio_material.name}")


# print("Registered radio materials:")
# for k, m in scene.radio_materials.items():
#     print(" -", k, m.name)



# =====================
# Engine init
# =====================
rx_fc = cfg["ue"]["rx_fc"]  # e.g. [1.5e10, 1.5e10, 3.5e9, 3.5e9]
fc_to_rx_idx = defaultdict(list)
for rx_idx, fc in enumerate(rx_fc):
    fc_to_rx_idx[fc].append(rx_idx)

print("Number of unique fc:", len(fc_to_rx_idx))
print("fc -> rx indices:", dict(fc_to_rx_idx))

engine = Engine(cfg, scene)

# =====================
# Run routes + save
# =====================
routes_dir = Path(cfg["route"]["folder"])
out_dir = Path(cfg["experiment"]["CQI_dir"])
out_dir.mkdir(parents=True, exist_ok=True)

# npz_files = sorted(routes_dir.glob("*.npz"))

# print("\n####################################################")
# print("starting trajectory processing...")
# for f in npz_files:
#     print(f"\nRunning {f.name}")
#     hp_list, sinr_db_list, sinr_lin_list, capacity_list = engine.run_from_file(f)

#     # 保险：很多时候这些是变长结构，直接存 array 会 shape 不一致
#     hp_arr = np.array(hp_list, dtype=object)
#     sinr_db_arr = np.array(sinr_db_list, dtype=object)
#     sinr_lin_arr = np.array(sinr_lin_list, dtype=object)
#     cap_arr = np.array(capacity_list, dtype=object)

#     out_path = out_dir / f"{f.stem}_result.npz"
#     np.savez_compressed(
#         out_path,
#         hp=hp_arr,
#         sinr_db=sinr_db_arr,
#         sinr_lin=sinr_lin_arr,
#         capacity=cap_arr,
#         route_file=f.name,
#     )

# print("\n####################################################")
# print(f"Done. Saved to: {out_dir}")


def is_valid_result(npz_path: Path) -> bool:
    """检查结果文件是否存在且字段齐全（可按需要加更严格检查）"""
    if not npz_path.exists():
        return False
    try:
        d = np.load(npz_path, allow_pickle=True)
        required = {"hp", "sinr_db", "sinr_lin", "capacity", "route_file"}
        if not required.issubset(set(d.files)):
            return False

        # 可选：更严格的 sanity check（避免空文件/半写入）
        sinr_db = d["sinr_db"]
        if sinr_db.size == 0:
            return False

        return True
    except Exception:
        return False


print("\n####################################################")
print("starting trajectory processing (resume enabled)...")

npz_files = sorted(routes_dir.glob("*.npz"))

for f in npz_files:
    out_path = out_dir / f"{f.stem}_result.npz"

    # ---- resume: skip if already done ----
    if is_valid_result(out_path):
        print(f"[SKIP] {f.name} -> already has {out_path.name}")
        continue

    print(f"\n[RUN ] {f.name}")

    try:
        hp_list, sinr_db_list, sinr_lin_list, capacity_list = engine.run_from_file(f)

        hp_arr = np.array(hp_list, dtype=object)
        sinr_db_arr = np.array(sinr_db_list, dtype=object)
        sinr_lin_arr = np.array(sinr_lin_list, dtype=object)
        cap_arr = np.array(capacity_list, dtype=object)

        # 原子写入：先写到临时文件，成功后再 rename，避免中途中断留下半文件
        tmp_path = out_dir / f"{f.stem}_result.tmp.npz"
        out_path = out_dir / f"{f.stem}_result.npz"

        np.savez_compressed(
            tmp_path,
            hp=hp_arr,
            sinr_db=sinr_db_arr,
            sinr_lin=sinr_lin_arr,
            capacity=cap_arr,
            route_file=f.name,
        )

        tmp_path.replace(out_path)


        print(f"[OK  ] saved -> {out_path.name}")

    except KeyboardInterrupt:
        print("\n[STOP] KeyboardInterrupt, exiting (safe).")
        raise
    except Exception as e:
        print(f"[FAIL] {f.name}: {type(e).__name__}: {e}")
        # 失败就继续下一个（也可以选择 raise）
        continue

print("\n####################################################")
print(f"Done. Results in: {out_dir}")

