"""
Isaac Lab DirectRL environment config for Chitrak quadruped locomotion.
Mirrors the structure of the mjlab_stock_gait env but using Isaac Lab APIs.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from chitrak_asset import CHITRAK_CFG


@configclass
class ChitrakSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
    )

    robot: ArticulationCfg = CHITRAK_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class ChitrakEnvCfg(DirectRLEnvCfg):
    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=4)

    # Scene
    scene: ChitrakSceneCfg = ChitrakSceneCfg(num_envs=4096, env_spacing=2.5)

    # RL settings
    episode_length_s: float = 20.0
    decimation: int = 4  # policy runs at 50 Hz (200 / 4)

    # Observation space: base lin vel (3) + base ang vel (3) + projected gravity (3)
    #                    + joint pos (12) + joint vel (12) + last action (12) = 45
    num_observations: int = 45
    # Action space: 12 joint position targets
    num_actions: int = 12

    # Action scale (radians)
    action_scale: float = 0.25

    # Default joint positions (used as action offset)
    default_joint_pos = {
        ".*_hip_roll_joint": 0.0,
        ".*_hip_pitch_joint": 0.5,
        ".*_knee_joint": -1.0,
    }

    # Reward scales
    class reward_cfg:
        lin_vel_xy_exp: float = 1.0
        ang_vel_z_exp: float = 0.5
        lin_vel_z_penalty: float = -2.0
        ang_vel_xy_penalty: float = -0.05
        torque_penalty: float = -0.0001
        dof_acc_penalty: float = -2.5e-7
        action_rate_penalty: float = -0.01
        feet_air_time: float = 0.5
        base_height_penalty: float = -1.0
        target_base_height: float = 0.22

    # Command ranges
    class command_cfg:
        lin_vel_x: tuple = (-1.0, 1.0)
        lin_vel_y: tuple = (-0.5, 0.5)
        ang_vel_z: tuple = (-1.0, 1.0)
        heading: tuple = (-math.pi, math.pi)
