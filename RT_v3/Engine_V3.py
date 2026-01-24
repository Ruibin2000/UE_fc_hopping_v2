# =======================
# TensorFlow / GPU setup
# =======================
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 用 GPU 0；设为 "" 用 CPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    print("GPU available:", gpus)
else:
    print("No GPU, using CPU")

print()

# =======================
# Core scientific stack
# =======================
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import json

# =======================
# Mitsuba / drjit 
# =======================
import mitsuba as mi
import drjit as dr

# =======================
# Sionna RT
# =======================
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

# =======================
# Utils / geometry
# =======================
from scipy.spatial.transform import Rotation as R
import sionnautils
from sionnautils.custom_scene import list_scenes, get_scene

# =======================
# Project-specific
# =======================
from UE_config_3 import UE_v3
from patch_pattern import Parch_Pattern

from collections import defaultdict




class Engine:
            
    ###############################################################################
    def __init__(self, cfg, scene):
        
        self.cfg = cfg
        # ------------------------------------------------------------------
        # register custom antenna patterns
        register_antenna_pattern("patch", lambda: Parch_Pattern("V"))

        # # ------------------------------------------------------------------
        # # create scene
        # scene_path, map_data = get_scene(scene_file)
        # for k, v in map_data.items():
        #     print(f'{k}: {v}')
            
        # self.scene = load_scene(scene_path,merge_shapes=True)
        
        # floor = self.scene.get('ground')
        # # print(f'Floor material: {floor.radio_material.name}')
        # floor.radio_material = ITURadioMaterial("itu_concrete",
        #                                 "concrete",
        #                                 thickness=0.01,
        #                                 color=(0.5, 0.5, 0.5))

        # self.scene.remove("itu_wet_ground")

        # for name, obj in self.scene.objects.items():
        #     print(f'{name:<15}{obj.radio_material.name}')
        self.scene = scene
            
            
        # ------------------------------------------------------------------
        # properties
        self.pattern_list = cfg["ue"]["rx_pattern"]
        rx_fc_list = cfg["ue"]["rx_fc"]   # [1.5e10, 1.5e10, 3.5e9, 3.5e9]
        
        self.fc_to_rx_idx = defaultdict(list)
        for rx_idx, fc in enumerate(rx_fc_list):
            self.fc_to_rx_idx[fc].append(rx_idx)
        
        self.B_list = cfg["ue"]["rx_bw"]
        self.Ptx_dBm = cfg["tx"]["power_dbm"]
        self.NF_dB = cfg["channel"]["NF_dB"]
        self.N0_dBm_per_Hz = cfg["channel"]["N0_dBm_per_Hz"]
        self.measure_time = cfg["channel"]["measure_time"]    # 50 ms
        
        self.rho_max = cfg["channel"]["rho_max"]
        self.alpha = cfg["channel"]["alpha"]
        
        self.rand_seed = cfg["channel"]["rand_seed"]
        np.random.seed(self.rand_seed)
        
        self.p_solver  = PathSolver()
        
        
        # ------------------------------------------------------------------
        # load tx info
        self.n_tx = cfg["tx"]["n_tx"]
        self.tx_loc =cfg["tx"]["pos"]
        self.look_at_loc = cfg["tx"]["look_at_pos"]
        
        # ------------------------------------------------------------------
        # create UEs
        self.n_ue = cfg["ue"]["n_ue"]
        self.ue_list = []

        for index_ue in range(self.n_ue):
            
            ue = UE_v3(scene = self.scene, id_ue= index_ue, cfg = cfg)
            self.ue_list.append(ue)
        
        # ------------------------------------------------------------------
        # create txs
        # for index_tx in range(cfg["tx"]["n_tx"]):

        self.scene.tx_array = PlanarArray(num_rows=cfg["tx"]["array_size"][0],
                                    num_cols=cfg["tx"]["array_size"][1],
                                    vertical_spacing=cfg["tx"]["vertical_spacing"],
                                    horizontal_spacing=cfg["tx"]["horizontal_spacing"],
                                    pattern=cfg["tx"]["pattern"],
                                    polarization=cfg["tx"]["polarization"])
            
        for index_tx in range(cfg["tx"]["n_tx"]):
            tx = Transmitter(name="tx-" + str(index_tx),
                                    position=cfg["tx"]["pos"][index_tx],
                                    display_radius=cfg["tx"]["display_radius"])

            # Add transmitter instance to scene
            self.scene.remove("tx-" + str(index_tx))
            self.scene.add(tx)

            tx.look_at(cfg["tx"]["look_at_pos"][index_tx])
            
        # ------------------------------------------------------------------
        # UE initial positions and orientations
        self.R_speed_level = cfg["ue"]["R_speed_level"]
        self.ue_yaw_init_list = np.deg2rad(np.array(cfg["ue"]["initial_orientation_deg"])[:,0])
        # self.ue_yaw_list = self.ue_yaw_init_list.copy()
        self.ue_pitch_init_list = np.deg2rad(np.array(cfg["ue"]["initial_orientation_deg"])[:,1])
        # self.ue_pitch_list = self.ue_pitch_init_list.copy()
        # self.ue_loc_init_list = np.array(cfg["ue"]["initial_pos"])
        # self.ue_loc_list = self.ue_loc_init_list.copy()
        
        self.reset()
        
    def reset(self):
        # ------------------------------------------------------------------
        self.ue_yaw_list = self.ue_yaw_init_list.copy()
        self.ue_pitch_list = self.ue_pitch_init_list.copy()
        # self.ue_loc_list = self.ue_loc_init_list.copy()
            
        # ------------------------------------------------------------------
        # initialize data record
        
    def set_ue_orientation(self, angles):
        
        self.ue_yaw_init_list = np.deg2rad(np.array(angles)[:,0])
        self.ue_pitch_init_list = np.deg2rad(np.array(angles)[:,1])
        
        self.reset()
        
    # ###############################################################################      
    def update_orientation(self, t):
        
        for index_ue in range(self.n_ue):

            step_yaw = np.random.uniform(- self.R_speed_level * np.pi * t * 2, self.R_speed_level * np.pi * t * 2)
            step_pitch = np.random.uniform(- self.R_speed_level * np.pi * t, self.R_speed_level * np.pi * t)
            self.ue_yaw_list[index_ue] += step_yaw
            self.ue_pitch_list[index_ue] += step_pitch
            
    ###############################################################################
    # helper function
    def db2lin(self,db_values):
        return np.power(10.0, db_values / 10.0)

    def lin2db(self,linear_values):
        return 10.0 * np.log10(linear_values)
    

    def merge_with_padding(self, a_list, ue_idx_list, n_rx, path_axis=4, pad_value=0):
        """
        Merge arrays by padding only the path dimension, then scatter batches
        into a unified output according to ue_idx_list.

        Parameters
        ----------
        a_list : list[np.ndarray]
            Arrays with same shape except at path_axis.
            Axis-0 must be global flattened RX index: (n_ue*n_rx, ...)
        ue_idx_list : list
            Nested list with structure [n_ue][n_group][local_rx_indices]
            Example (n_ue=2, n_group=2, n_rx=4):
                [
                [[0,1], [2,3]],
                [[0,1], [2,3]],
                ]
            Group g corresponds to a_list[g].
        n_rx : int
            Number of RX per UE (used to convert local -> global index).
        path_axis : int
            Path dimension axis to pad.
        pad_value : scalar
            Padding fill value.

        Returns
        -------
        out : np.ndarray
            Unified array with path dimension padded to max_P.
        """
        assert len(a_list) > 0, "a_list must be non-empty"

        n_ue = len(ue_idx_list)
        n_group = len(ue_idx_list[0])
        assert len(a_list) == n_group, "len(a_list) must equal number of groups per UE"

        # ---- build fc_index_list (global flattened indices) from ue_idx_list ----
        fc_index_list = []
        for g in range(n_group):
            global_idx = []
            for ue in range(n_ue):
                if len(ue_idx_list[ue]) != n_group:
                    raise ValueError("All UEs must have the same number of groups")

                local = np.asarray(ue_idx_list[ue][g], dtype=int)
                global_idx.extend((ue * n_rx + local).tolist())
            fc_index_list.append(np.asarray(global_idx, dtype=int))

        # ---- original logic (now uses fc_index_list) ----
        ndim = a_list[0].ndim
        if path_axis < 0:
            path_axis = ndim + path_axis
        if not (0 <= path_axis < ndim):
            raise ValueError(f"Invalid path_axis={path_axis} for ndim={ndim}")

        base_shape = list(a_list[0].shape)
        N = base_shape[0]  # should be n_ue*n_rx

        # sanity: check N matches n_ue*n_rx
        if N != n_ue * n_rx:
            raise ValueError(f"Axis-0 size mismatch: a_list[0].shape[0]={N} != n_ue*n_rx={n_ue*n_rx}")

        # check consistency except path_axis
        for k, a in enumerate(a_list):
            if a.ndim != ndim:
                raise ValueError(f"a_list[{k}] ndim={a.ndim} != {ndim}")
            if a.shape[0] != N:
                raise ValueError(f"a_list[{k}] axis-0={a.shape[0]} != {N}")
            for ax in range(ndim):
                if ax == path_axis:
                    continue
                if a.shape[ax] != base_shape[ax]:
                    raise ValueError(
                        f"a_list[{k}] shape mismatch at axis {ax}: {a.shape[ax]} != {base_shape[ax]} "
                        f"(only path_axis may differ)"
                    )

        max_P = max(a.shape[path_axis] for a in a_list)

        out_shape = list(base_shape)
        out_shape[path_axis] = max_P
        dtype = np.result_type(*a_list)

        out = np.full(out_shape, pad_value, dtype=dtype)

        # scatter fill
        for a, idx in zip(a_list, fc_index_list):
            P = a.shape[path_axis]
            slicer = [slice(None)] * ndim
            slicer[0] = idx
            slicer[path_axis] = slice(0, P)
            out[tuple(slicer)] = a[tuple(slicer)]

        return out



    
    ###############################################################################
    # run from file
    def run_from_file(self, routes_file):
        
        with open(routes_file, 'r') as f:
            loaded_routes = json.load(f)
            
        loaded_routes = np.array(loaded_routes)
            
            
        N_measure = loaded_routes.shape[1]
        
        self.reset()
        
        hp_total = []
        sinr_total = []
        sinr_db_total = []
        
        capacity_total = []
        
        # outer loop: update UE positions and orientations
        for index_measure in trange(N_measure):
            
            self.ue_loc_list = loaded_routes[:,index_measure,:].tolist()
            self.update_orientation(self.measure_time)
            
            self.run_RT()
            self.compute_sinr()
            self.compute_capacity()
            
            hp_total.append(self.hp_lin.tolist())
            sinr_total.append(self.sinr_lin_all_tx.tolist())
            sinr_db_total.append(self.sinr_db_all_tx.tolist())
            capacity_total.append(self.capacity_all_tx.tolist())
          
            
        return hp_total, sinr_db_total, sinr_total, capacity_total

    
    # def run_capacity_map(self, coords_array):
        
    #     x_dim, y_dim, _ = coords_array.shape    
        
    #     capacity_map = np.zeros((x_dim, y_dim, self.cfg["tx"]["n_tx"], self.cfg["ue"]["n_rx"]))
        
      
    #     pbar = tqdm(total=x_dim * y_dim, desc="Scanning grid", unit="pt")
    #     for idx_x in range(x_dim):
    #         for idx_y in range(y_dim):
    #             self.ue_loc_list = [coords_array[idx_x, idx_y, :].tolist()]


    #             self.run_RT()
    #             self.compute_sinr()
    #             self.compute_capacity()
    #             capacity_map[idx_x, idx_y] = self.capacity_all_tx.reshape(self.cfg["tx"]["n_tx"], self.cfg["ue"]["n_rx"])

    #             pbar.update(1)
    #     pbar.close()
                
    #     return capacity_map
    
    def run_capacity_map(self, coords_array):

        x_dim, y_dim, _ = coords_array.shape    
        n_tx = self.cfg["tx"]["n_tx"]
        n_rx = self.cfg["ue"]["n_rx"]

        capacity_map = np.zeros((x_dim, y_dim, n_tx, n_rx), dtype=float)

        pbar = tqdm(total=x_dim * y_dim, desc="Scanning grid", unit="pt")
        for idx_x in range(x_dim):
            for idx_y in range(y_dim):
                self.ue_loc_list = [coords_array[idx_x, idx_y, :].tolist()]

                self.run_RT()
                self.compute_sinr()
                self.compute_capacity()

                # capacity_all_tx shape: (n_ue, n_rx, 1, n_tx, 1, 1)
                cap = self.capacity_all_tx[0, :, 0, :, 0, 0]   # (n_rx, n_tx)
                capacity_map[idx_x, idx_y, :, :] = cap.T       # (n_tx, n_rx)

                pbar.update(1)

        pbar.close()
        return capacity_map

    
    
    def run_rotation_map(self, step_size):
        
        self.ue_loc_list = self.cfg["ue"]["initial_pos"]
        
        yaw_deg_list = list(range(-180, 181, step_size))
        yaw_rad_list = np.deg2rad(np.array(yaw_deg_list)).tolist()
        x_dim = len(yaw_deg_list)
        
        
        pitch_deg_list = list(range(-90, 91, 10))
        pitch_rad_list = np.deg2rad(np.array(pitch_deg_list)).tolist()
        y_dim = len(pitch_deg_list)
        
        rotation_sinr_map = np.zeros((x_dim, y_dim, self.cfg["ue"]["n_rx"]))
        
        pbar = tqdm(total=x_dim * y_dim, desc="Scanning grid", unit="pt")
        for idx_x in range(x_dim):
            for idx_y in range(y_dim):
                
                
                for index_ue in range(self.n_ue):
                    
                    self.ue_yaw_list[index_ue] = yaw_rad_list[idx_x]
                    self.ue_pitch_list[index_ue] = pitch_rad_list[idx_y]
                    
                self.run_RT()
                self.compute_sinr()

                
                
                
                rotation_sinr_map[idx_x, idx_y] = self.sinr_db_all_tx.reshape(self.cfg["ue"]["n_rx"],)

                pbar.update(1)
        pbar.close()
        
        return rotation_sinr_map



                
    def run_RT(self):
        a_list = []
        
        fc_index_list = []

        for index_ue in range(self.n_ue):
            self.ue_list[index_ue].set_location(self.ue_loc_list[index_ue])
            self.ue_list[index_ue].set_orientation(self.ue_yaw_list[index_ue], self.ue_pitch_list[index_ue], 0.0)
                
        for fc, idx_list in self.fc_to_rx_idx.items():
            # print(fc, idx_list)
            
            self.scene.frequency =  fc
            
            if np.abs(fc - 15.e9) < 1e-6:
                self.scene.rx_array = PlanarArray(num_rows=self.cfg["ue"]["rx_array_size"][0],
                                    num_cols=self.cfg["ue"]["rx_array_size"][1],
                                    vertical_spacing=self.cfg["ue"]["vertical_spacing"],
                                    horizontal_spacing=self.cfg["ue"]["horizontal_spacing"],
                                    pattern=self.cfg["ue"]["rx_pattern"][idx_list[0]])
                  
            elif np.abs(fc - 3.5e9) < 1e-6:
                self.scene.rx_array = PlanarArray(num_rows=self.cfg["ue"]["rx_array_size"][0],
                                    num_cols=self.cfg["ue"]["rx_array_size"][1],
                                    vertical_spacing=self.cfg["ue"]["vertical_spacing"],
                                    horizontal_spacing=self.cfg["ue"]["horizontal_spacing"],
                                    pattern=self.cfg["ue"]["rx_pattern"][idx_list[0]],
                                    polarization=self.cfg["ue"]["polarization"])
                
            paths = self.p_solver(scene=self.scene,
                     max_depth=self.cfg["rt"]["max_depth"],
                     los=self.cfg["rt"]["los"],
                     specular_reflection=self.cfg["rt"]["specular_reflection"],
                     diffraction=self.cfg["rt"]["diffraction"],
                     edge_diffraction=self.cfg["rt"]["edge_diffraction"],
                     refraction=self.cfg["rt"]["refraction"],
                     diffuse_reflection=self.cfg["rt"]["diffuse_reflection"])
            
            a, tau = paths.cir(normalize_delays=False, out_type="numpy")
            
            a_list.append(a)
            fc_index_list.append(idx_list)
        
        ue_idx_list = []   
        for index_ue in range(self.n_ue):
            ue_idx_list.append(fc_index_list)
            
        a_merged = self.merge_with_padding(
                        a_list=a_list,
                        ue_idx_list=ue_idx_list,
                        n_rx=self.cfg["ue"]["n_rx"],
                        path_axis=4,
                        pad_value=0
                    )
        n_rx = self.cfg["ue"]["n_rx"]

        self.a_merged_ue_rx = a_merged.reshape(
            self.n_ue,
            n_rx,
            *a_merged.shape[1:]
        )
        
        # self.a_merged_ue_rx

        self.hp_lin = np.sum(np.abs(self.a_merged_ue_rx)**2, axis=5)
        self.hp_db = self.lin2db(self.hp_lin)
        

    # -----------------------------
    # Power utilities
    # -----------------------------
    def dbm_to_w(self, p_dbm):
        """dBm -> Watt"""
        p_dbm = np.asarray(p_dbm, dtype=np.float64)
        return 10.0 ** ((p_dbm - 30.0) / 10.0)

    def noise_w_per_rx(self):
        """
        Noise power per RX in Watt:
        N_dBm = N0(dBm/Hz) + 10log10(B) + NF
        """
        B = np.asarray(self.B_list, dtype=np.float64)  # (n_rx,)
        N_dBm = float(self.N0_dBm_per_Hz) + 10.0*np.log10(B) + float(self.NF_dB)
        return self.dbm_to_w(N_dBm)  # (n_rx,)
    
    # -----------------------------
    # SINR computation (store ALL tx cases)
    # -----------------------------
    def compute_sinr(self):
        """
        Compute SINR and store ONLY 'all-tx-as-serving' results.

        Requires:
          self.hp_lin with shape:
            (n_ue, n_rx, n_rx_array, n_tx, n_tx_array, n_time)

        Stores:
          self.sinr_lin_all_tx :
            (n_ue, n_rx, n_rx_array, n_tx, n_tx_array, n_time)
        """
        if not hasattr(self, "hp_lin"):
            raise RuntimeError("hp_lin not found. Call run_RT() first.")

        case = self.cfg["channel"]["case"].upper()

        hp = np.asarray(self.hp_lin, dtype=np.float64)
        U, R, Rarr, T, Tarr, Time = hp.shape

        # ----------------------------------
        # SISO case (current implementation)
        # ----------------------------------
        if case == "SISO":

            if Rarr != 1 or Tarr != 1:
                raise ValueError(
                    f"SISO case expects n_rx_array == n_tx_array == 1, "
                    f"got Rarr={Rarr}, Tarr={Tarr}"
                )

            # TX power -> Watt
            Ptx_W = self.dbm_to_w(self.Ptx_dBm)  # (n_tx,)

            # Received power per link
            # hp: (U,R,1,T,1,Time)
            Prx_W = hp * Ptx_W[None, None, None, :self.n_tx, None, None]

            # Noise per RX
            N_W = self.noise_w_per_rx()                      # (R,)
            N = N_W[None, :, None, None, None, None]         # broadcast

            # Total received power across all TX
            total = Prx_W.sum(axis=(3, 4), keepdims=True)    # (U,R,1,1,1,Time)

            # SINR for each TX treated as desired
            sinr_all_tx = Prx_W / (total - Prx_W + N)

            self.sinr_lin_all_tx = sinr_all_tx
            self.sinr_db_all_tx = 10.0 * np.log10(np.maximum(sinr_all_tx, 1e-300))
            
            
            return

        # ----------------------------------
        # MIMO / other cases (future)
        # ----------------------------------
        else:
            raise NotImplementedError(
                f"SINR for channel case '{case}' is not implemented yet. "
                "This is a placeholder for future MIMO extensions."
            )

    def compute_capacity(self):

        sinr_lin_masked = np.where(self.sinr_db_all_tx < -5, 0.0, self.sinr_lin_all_tx)
        
        B_rx = np.asarray(self.B_list).reshape(
            1,            # n_ue
            -1,           # n_rx
            1,            # n_rx_array
            1,            # n_tx
            1,            # n_tx_array
            1             # n_t
        )
        
        se = self.alpha * np.log2(1.0 + sinr_lin_masked)
        se = np.minimum(se, self.rho_max)

        self.capacity_all_tx = B_rx * se / 1e6


        # self.capacity_all_tx = self.B_list * np.minimum(self.alpha * np.log2(1 + sinr_lin_masked), self.rho_max) / 1e6