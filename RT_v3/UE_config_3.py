# Import or install Sionna
try:
    import sionna.rt
except ImportError as e:
    
    import os
    os.system("pip install sionna-rt")
    import sionna.rt

from sionna.rt import load_scene, PlanarArray, Transmitter, Receiver, Camera,\
                      PathSolver, ITURadioMaterial, SceneObject


import numpy as np

import drjit as dr
import mitsuba as mi
import gym

from scipy.spatial.transform import Rotation as Rot



class UE_v3:
    def __init__(self, scene, id_ue, cfg):
        self.scene = scene
        self.id = id_ue
        self.scaling = cfg["ue"]["scaling"][id_ue]
        

        self.ue_material = ITURadioMaterial(f"UE-material-{id_ue}",
                                        cfg["ue"]["body_material"],
                                        thickness=0.1,
                                        color=(0.8, 0, 0))
        
        
        # Create a new object with the given parameters
        self.body = SceneObject(fname=cfg["ue"]["body_model"],
                               name=f"ue-{self.id}",
                               radio_material=self.ue_material)
        
        
        self.scene.remove(f"ue-{self.id}")
        self.scene.edit(add = self.body)
        self.body.scaling = self.scaling

        self.reset(cfg)

    def reset(self, cfg):
        self.center_pos = cfg["ue"]["initial_pos"][self.id]
        # self.center_pos = np.asarray(cfg["ue"]["center_pos"][self.id], dtype=np.float64).reshape(3)

        
        # Configure RXs
        self.n_rx = cfg["ue"]["n_rx"]
        self.rx_loc_pos_list = []
        self.rx_loc_oritation_list = []
        self.rx_pattern_list = []
        
        for index_rx in range(self.n_rx):
            self.rx_loc_pos_list.append(np.array(cfg["ue"]["rx_loc_pos"][index_rx]) * self.scaling)
            self.rx_loc_oritation_list.append(np.deg2rad(np.array(cfg["ue"]["rx_loc_orientation_deg"][index_rx])))
            # self.rx_pattern_list.append(cfg["ue"]["rx_pattern"][index_rx])
            
        # Create RXs
        self.rx_list = []
        
        for index_rx in range(self.n_rx):
            rx = Receiver(name=f"ue{self.id}-rx-{index_rx}",
                        position=self.center_pos + self.rx_loc_pos_list[index_rx],
                        orientation=self.rx_loc_oritation_list[index_rx],
                        display_radius=cfg["ue"]["rx_display_radius"][index_rx])
            
            self.scene.remove(f"ue{self.id}-rx-{index_rx}")
            self.scene.add(rx)
            
            self.rx_list.append(rx)
            
        # Reset the position and orientation of the object
        self.set_location(self.center_pos)
        

        self.body.orientation = np.array([0, 0, 0])

        for index_rx in range(self.n_rx):
            self.rx_list[index_rx].orientation = self.rx_loc_oritation_list[index_rx]

         
                        
    def set_location(self,displacement):
        # Update the position of the object
        # print(displacement)
        self.body.position = displacement
        self.center_pos = displacement

        for index_rx in range(self.n_rx):
            self.rx_list[index_rx].position = self.rx_loc_pos_list[index_rx] + self.center_pos
            
            
    def R_matrix (self, yaw, pitch, roll):
        # Rotation about X (roll)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])

        # Rotation about Y (pitch)
        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # Rotation about Z (yaw)
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combined rotation: ZYX (yaw → pitch → roll)
        R = Rz @ Ry @ Rx

        return R


    # def set_orientation(self, alpha, beta, gamma):

    #     # self.set_location(self.center_pos)


    #     self.body.orientation = np.array([alpha, beta, gamma])
    #     for index_rx in range(self.n_rx):
    #         self.rx_list[index_rx].orientation = self.rx_loc_oritation_list[index_rx] + np.array([alpha, beta, gamma])
    #     R = self.R_matrix(alpha, beta, gamma)
        

    #     for index_rx in range(self.n_rx):
    #         self.rx_list[index_rx].position = R @ self.rx_loc_pos_list[index_rx] + self.center_pos

    def set_orientation(self, alpha, beta, gamma):
        # UE 姿态 (ZYX: yaw, pitch, roll)  你这里 gamma 常为 0
        self.body.orientation = np.array([alpha, beta, gamma])

        # UE 旋转矩阵
        R_ue = self.R_matrix(alpha, beta, gamma)

        # 位置：只需要 UE 旋转
        for index_rx in range(self.n_rx):
            self.rx_list[index_rx].position = R_ue @ self.rx_loc_pos_list[index_rx] + self.center_pos

        # 朝向：必须做旋转组合（不能角度相加）
        for index_rx in range(self.n_rx):
            yaw_l, pitch_l, roll_l = self.rx_loc_oritation_list[index_rx]  # (yaw,pitch,roll) in rad

            R_local = self.R_matrix(yaw_l, pitch_l, roll_l)
            R_world = R_ue @ R_local

            # 把 R_world 转回 ZYX 欧拉角（yaw,pitch,roll）
            eul = Rot.from_matrix(R_world).as_euler("ZYX", degrees=False)

            # eul 顺序就是 (yaw, pitch, roll) 与你 config 对齐
            self.rx_list[index_rx].orientation = eul
