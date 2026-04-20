# =====================
# TensorFlow / GPU (MUST set env BEFORE importing TF)
# =====================
import os
import logging
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

num_workers = 4
work_id = 3  


# 只在没设置时才默认用 GPU 0；想强制 CPU：os.environ["CUDA_VISIBLE_DEVICES"] = ""
if os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{work_id % 2}"


import tensorflow as tf

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
from tqdm import tqdm
import yaml

mi.set_log_level(mi.LogLevel.Error)   # 只显示 Error，不显示 Warn


# ============================================================
# Logger helpers
# ============================================================
def setup_logger(work_id):
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"CQI_{work_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("RT_CQI")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # 防止重复添加 handler
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.info("Logger initialized")
    logger.info(f"Log file: {log_file}")

    return logger, log_file


def log_dict(logger, title, d):
    logger.info(title)
    for k, v in d.items():
        logger.info(f"{k:20s}: {v}")
    logger.info("=" * 40)


# ============================================================
# Config helpers
# ============================================================
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


def load_cfg(cfg_path="config.yaml"):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return normalize_config(cfg)


# ============================================================
# Result validation
# ============================================================
def is_valid_result(npz_path: Path, logger=None) -> bool:
    """检查结果文件是否存在且字段齐全"""
    if not npz_path.exists():
        return False

    try:
        d = np.load(npz_path, allow_pickle=True)
        required = {"hp", "sinr_db", "sinr_lin", "capacity", "route_file"}
        if not required.issubset(set(d.files)):
            if logger is not None:
                logger.warning(f"Invalid result file (missing keys): {npz_path.name}")
            return False

        # 更严格的 sanity check
        sinr_db = d["sinr_db"]
        if sinr_db.size == 0:
            if logger is not None:
                logger.warning(f"Invalid result file (empty sinr_db): {npz_path.name}")
            return False

        return True

    except Exception:
        if logger is not None:
            logger.exception(f"Failed to validate result file: {npz_path}")
        return False


# ============================================================
# Scene loader
# ============================================================
def load_and_prepare_scene(cfg, logger):
    map_data = None

    if cfg["scene"]["name"] != "free_space":
        # 如果你之后想严格按 config 走，可以改成 get_scene(cfg["scene"]["name"])
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
                logger.info(f"{name:<20}{obj.radio_material.name}")
            except Exception:
                logger.info(f"{name:<20}<no radio material>")
        logger.info("=" * 40)

    else:
        scene = load_scene()
        logger.info("Loaded free_space scene")

    return scene, map_data


# ============================================================
# Main
# ============================================================
def main():
    
    global num_workers
    global work_id
    logger, log_file = setup_logger(work_id= work_id)
    start_total = time.time()

    # ---------------------
    # TF / GPU info
    # ---------------------
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logger.info(f"GPU available: {gpus}")
    else:
        logger.info("No GPU, using CPU")

    logger.info(f"TF GPUs: {gpus}")
    logger.info(f"Mitsuba variant: {mi.variant()}")
    logger.info(f"CUDA_VISIBLE_DEVICES = {os.getenv('CUDA_VISIBLE_DEVICES')}")

    # ---------------------
    # Load config
    # ---------------------
    cfg = load_cfg("config_arr8_s3_medium.yaml")

    log_dict(logger, "===== Route parameters =====", cfg["route"])
    log_dict(logger, "===== Motion parameters =====", cfg["motion"])

    # ---------------------
    # Load scene
    # ---------------------
    scene, map_data = load_and_prepare_scene(cfg, logger)

    # ---------------------
    # Engine init
    # ---------------------
    rx_fc = cfg["ue"]["rx_fc"]  # e.g. [1.5e10, 1.5e10, 3.5e9, 3.5e9]
    fc_to_rx_idx = defaultdict(list)
    for rx_idx, fc in enumerate(rx_fc):
        fc_to_rx_idx[fc].append(rx_idx)

    logger.info(f"Number of unique fc: {len(fc_to_rx_idx)}")
    logger.info(f"fc -> rx indices: {dict(fc_to_rx_idx)}")

    engine = Engine(cfg, scene)

    # ---------------------
    # Run routes + save
    # ---------------------
    routes_dir = Path(cfg["route"]["folder"])
    out_dir = Path(cfg["experiment"]["CQI_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Routes dir: {routes_dir}")
    logger.info(f"Output dir: {out_dir}")

    # npz_files = sorted(routes_dir.glob("*.npz"))
    
    npz_files = sorted(routes_dir.glob("*.npz"))

    npz_files = npz_files[work_id::num_workers]
    logger.info(f"Found {len(npz_files)} route files")

    logger.info("####################################################")
    logger.info("starting trajectory processing (resume enabled)...")

    stats = {
        "skip": 0,
        "run": 0,
        "success": 0,
        "fail": 0,
    }

    for f in tqdm(npz_files, desc="Processing routes"):
        out_path = out_dir / f"{f.stem}_result.npz"

        # ---- resume: skip if already done ----
        if is_valid_result(out_path, logger=logger):
            stats["skip"] += 1
            logger.info(f"[SKIP] {f.name} -> already has {out_path.name}")
            continue

        stats["run"] += 1
        logger.info(f"[RUN ] {f.name}")
        t0 = time.time()

        try:
            hp_list, sinr_db_list, sinr_lin_list, capacity_list = engine.run_from_file(f)

            hp_arr = np.array(hp_list, dtype=object)
            sinr_db_arr = np.array(sinr_db_list, dtype=object)
            sinr_lin_arr = np.array(sinr_lin_list, dtype=object)
            cap_arr = np.array(capacity_list, dtype=object)

            # 原子写入：先写到临时文件，成功后再 rename，避免中途中断留下半文件
            tmp_path = out_dir / f"{f.stem}_result.tmp.npz"

            np.savez_compressed(
                tmp_path,
                hp=hp_arr,
                sinr_db=sinr_db_arr,
                sinr_lin=sinr_lin_arr,
                capacity=cap_arr,
                route_file=f.name,
            )

            tmp_path.replace(out_path)

            dt = time.time() - t0
            stats["success"] += 1

            logger.info(
                f"[OK  ] saved -> {out_path.name} | "
                f"time={dt:.2f}s | "
                f"hp.shape={hp_arr.shape} | "
                f"sinr_db.shape={sinr_db_arr.shape} | "
                f"sinr_lin.shape={sinr_lin_arr.shape} | "
                f"capacity.shape={cap_arr.shape}"
            )

        except KeyboardInterrupt:
            logger.warning("[STOP] KeyboardInterrupt, exiting (safe).")
            raise

        except Exception:
            stats["fail"] += 1
            logger.exception(f"[FAIL] {f.name}")
            continue

    total_time = time.time() - start_total

    logger.info("####################################################")
    logger.info(f"Done. Results in: {out_dir}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Total time: {total_time:.2f}s ({total_time/60.0:.2f} min)")
    logger.info(f"Summary: {stats}")


if __name__ == "__main__":
    main()