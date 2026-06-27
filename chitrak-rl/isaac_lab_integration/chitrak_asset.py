"""
Chitrak quadruped ArticulationCfg for Isaac Lab.

Run convert_urdf_to_usd.py first to generate the USD file, then import CHITRAK_CFG here.
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg

USD_PATH = "/workspace/isaaclab/source/chitrak_integration/usd_output/chitrak.usd"

CHITRAK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.25),
        joint_pos={
            ".*_hip_roll_joint": 0.0,
            # FR/BR pitch: limits [0, 3.14]
            "fr_hip_pitch_joint": 0.5,
            "br_hip_pitch_joint": 0.5,
            # FL/BL pitch: limits [-3.14, 0]
            "fl_hip_pitch_joint": -0.5,
            "bl_hip_pitch_joint": -0.5,
            # FR/BR knee: limits [-3.14, 0]
            "fr_knee_joint": -1.0,
            "br_knee_joint": -1.0,
            # FL/BL knee: limits [0, 3.14]
            "fl_knee_joint": 1.0,
            "bl_knee_joint": 1.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit=2.5,
            saturation_effort=2.5,
            velocity_limit=8.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
