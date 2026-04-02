# =====================
# TensorFlow / GPU (MUST set env BEFORE importing TF)
# =====================
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print("GPU available:" if gpus else "No GPU, using CPU", gpus)

# =====================
# Scientific stack
# =====================
import numpy as np

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
import json
from collections import defaultdict

mi.set_log_level(mi.LogLevel.Error)


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

# =====================
# Engine init
# =====================
rx_fc = cfg["ue"]["rx_fc"]
fc_to_rx_idx = defaultdict(list)
for rx_idx, fc in enumerate(rx_fc):
    fc_to_rx_idx[fc].append(rx_idx)

print("Number of unique fc:", len(fc_to_rx_idx))
print("fc -> rx indices:", dict(fc_to_rx_idx))

engine = Engine(cfg, scene)

# =====================
# Process v2 routes
# =====================
routes_base_dir = Path(cfg["route"]["folder"])
out_dir = Path(cfg["experiment"]["CQI_dir"])
out_dir.mkdir(parents=True, exist_ok=True)

# 三条v2生成的路径
v2_routes = [
    ("spin_level_v2_0.0", 0.0),
    ("spin_level_v2_0.5", 0.5),
    ("spin_level_v2_1.0", 1.0),
]

print("\n" + "="*70)
print("【处理v2生成的三条路径】")
print("="*70 + "\n")

results_summary = {}

for route_folder, spin_level in v2_routes:
    route_subdir = routes_base_dir / route_folder
    route_file = route_subdir / f"route_v2_spin_{spin_level:.1f}.json"
    
    if not route_file.exists():
        print(f"[SKIP] {route_file.name} - 文件不存在")
        continue
    
    print(f"\n{'='*70}")
    print(f"R_speed_level = {spin_level}")
    print(f"{'='*70}")
    print(f"路径文件: {route_file}")
    
    # ---- 从JSON加载路由 ----
    try:
        with open(route_file, 'r') as f:
            route_data = json.load(f)
    except Exception as e:
        print(f"[FAIL] JSON加载失败: {e}")
        continue
    
    positions = np.array(route_data["positions"], dtype=np.float32)  # (n_steps, 3)
    rotations = np.array(route_data["rotations"], dtype=np.float32)  # (n_steps, 3)
    
    print(f"步数: {len(positions)}")
    print(f"起点: {positions[0]}")
    print(f"终点: {positions[-1]}")
    
    # ---- 调整维度为 (n_ue, n_steps, 3) - 这里n_ue=1 ----
    positions = np.expand_dims(positions, axis=0)  # (1, n_steps, 3)
    rotations = np.expand_dims(rotations, axis=0)  # (1, n_steps, 3)
    
    print(f"位置尺寸: {positions.shape}")
    print(f"旋转尺寸: {rotations.shape}")
    
    # ---- 创建临时NPZ文件用于Engine.run_from_file ----
    tmp_npz_path = route_subdir / f"route_v2_spin_{spin_level:.1f}_temp.npz"
    try:
        np.savez(
            tmp_npz_path,
            positions=positions,
            rotations=rotations,
        )
        
        print(f"临时NPZ: {tmp_npz_path.name}")
        
        # ---- 运行引擎 ----
        print("运行引擎...")
        hp_list, sinr_db_list, sinr_lin_list, capacity_list = engine.run_from_file(tmp_npz_path)
        
        # ---- 保存结果 ----
        hp_arr = np.array(hp_list, dtype=object)
        sinr_db_arr = np.array(sinr_db_list, dtype=object)
        sinr_lin_arr = np.array(sinr_lin_list, dtype=object)
        cap_arr = np.array(capacity_list, dtype=object)
        
        result_filename = f"v2_spin_{spin_level:.1f}_noBuilding_result.npz"
        result_path = out_dir / result_filename
        
        np.savez_compressed(
            result_path,
            hp=hp_arr,
            sinr_db=sinr_db_arr,
            sinr_lin=sinr_lin_arr,
            capacity=cap_arr,
            spin_level=spin_level,
            route_file=str(route_file.name),
        )
        
        print(f"✅ 结果已保存: {result_path}")
        
        # 统计摘要
        sinr_db_flat = np.concatenate([s.flatten() for s in sinr_db_arr])
        capacity_flat = np.concatenate([c.flatten() for c in cap_arr])
        
        results_summary[spin_level] = {
            "sinr_db_mean": float(np.mean(sinr_db_flat)),
            "sinr_db_std": float(np.std(sinr_db_flat)),
            "sinr_db_min": float(np.min(sinr_db_flat)),
            "sinr_db_max": float(np.max(sinr_db_flat)),
            "capacity_mean": float(np.mean(capacity_flat)),
            "capacity_std": float(np.std(capacity_flat)),
        }
        
    except Exception as e:
        print(f"[FAIL] 处理失败: {type(e).__name__}: {e}")
        continue
    finally:
        # 清理临时文件
        if tmp_npz_path.exists():
            tmp_npz_path.unlink()
            print(f"临时文件已清理: {tmp_npz_path.name}")

# =====================
# Summary
# =====================
print("\n" + "="*70)
print("【处理结果摘要】")
print("="*70 + "\n")

print("输出目录:", out_dir)

if results_summary:
    print("\n| 指标 | R_speed=0.0 | R_speed=0.5 | R_speed=1.0 |")
    print("|------|-------------|-------------|-------------|")
    
    for metric in ["sinr_db_mean", "sinr_db_std", "capacity_mean"]:
        print(f"| {metric:20s} |", end="")
        for spin_level in [0.0, 0.5, 1.0]:
            if spin_level in results_summary:
                val = results_summary[spin_level].get(metric, "--")
                print(f" {val:11.4f} |" if isinstance(val, (int, float)) else f" {val:11s} |", end="")
            else:
                print(f" {'--':11s} |", end="")
        print()
    
    print("\n详细数据:")
    for spin_level, stats in sorted(results_summary.items()):
        print(f"\n  R_speed={spin_level}:")
        for k, v in stats.items():
            print(f"    {k}: {v:.6f}")

print("\n" + "="*70)
print("✅ 完成！")
print("="*70)
