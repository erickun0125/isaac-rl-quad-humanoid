# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""Configuration for Unitree robots."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR


G129_CFG_WITH_DEX3_BASE_FIX = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"sequor_g1/envs/usd/robots/g1-29dof-dex3-base-fix-usd/g1_29dof_with_dex3_base_fixed.usd",
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
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,

        ),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # legs joints
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            
            # waist joints
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            
            # arms joints
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            
            # fingers joints
            "left_hand_index_0_joint": 0.0,
            "left_hand_middle_0_joint": 0.0,
            "left_hand_thumb_0_joint": 0.0,
            "left_hand_index_1_joint": 0.0,
            "left_hand_middle_1_joint": 0.0,
            "left_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.0,
            
            "right_hand_index_0_joint": 0.0,
            "right_hand_middle_0_joint": 0.0,
            "right_hand_thumb_0_joint": 0.0,
            "right_hand_index_1_joint": 0.0,
            "right_hand_middle_1_joint": 0.0,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,

    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint", 
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint", 
                ".*_knee_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],  
            effort_limit=1000.0,  # set a large torque limit
            velocity_limit=0.0,   # set the velocity limit to 0
            stiffness={
                "waist_yaw_joint": 400.0,
                "waist_roll_joint": 400.0,
                "waist_pitch_joint": 400.0
            },
            damping={
                "waist_yaw_joint": 4.0,
                "waist_roll_joint": 4.0,
                "waist_pitch_joint": 4.0
            },
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=None,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=None,
            damping=None,
            # armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint"
            ],
            effort_limit=None,
            velocity_limit=None,
             stiffness={  # increase the stiffness (kp)
                 ".*_shoulder_.*_joint": 300.0,
                 ".*_elbow_joint": 400.0,
                 ".*_wrist_.*_joint": 400.0,
            },
             damping={    # increase the damping (kd)
                 ".*_shoulder_.*_joint": 3.0,
                 ".*_elbow_joint": 2.5,
                 ".*_wrist_.*_joint": 2.5,
             },
            armature=None,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hand_index_.*_joint",
                ".*_hand_middle_.*_joint",
                ".*_hand_thumb_.*_joint"
            ],
            effort_limit=100,
            velocity_limit=None,
            # stiffness=None,
            # damping=None,
            stiffness={  # increase the stiffness (kp)
                ".*_hand_index_.*_joint": 400.0,
                ".*_hand_middle_.*_joint": 400.0,
                ".*_hand_thumb_.*_joint": 400.0,
            },
            damping={    # increase the damping (kd)
               ".*_hand_index_.*_joint": 4.0,
               ".*_hand_middle_.*_joint": 4.0,
               ".*_hand_thumb_.*_joint": 4.0,
            },
            armature=None,
        ),
    },
)




G129_CFG_WITH_DEX1_BASE_FIX = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"usd/robots/g1-29dof-dex1-base-fix-usd/g1_29dof_with_dex1_base_fix.usd",
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
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4
        ),

    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            # legs joints
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            
            # waist joints
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            
            # arms joints
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            
            # fingers joints
            "left_hand_Joint1_1": 0.0,
            "left_hand_Joint2_1": 0.0,
            "right_hand_Joint1_1": 0.0,
            "right_hand_Joint2_1": 0.0,
            
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint", 
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint", 
                ".*_knee_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=[
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint"
            ],  
            effort_limit=1000.0,  # set a large torque limit
            velocity_limit=0.0,   # set the velocity limit to 0
            stiffness={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            damping={
                "waist_yaw_joint": 10000.0,
                "waist_roll_joint": 10000.0,
                "waist_pitch_joint": 10000.0
            },
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit=None,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=None,
            damping=None,
            # armature=0.001,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint"
            ],
            effort_limit=None,
            velocity_limit=None,
             stiffness={  # increase the stiffness (kp)
                 ".*_shoulder_.*_joint": 300.0,
                 ".*_elbow_joint": 400.0,
                 ".*_wrist_.*_joint": 400.0,
            },
             damping={    # increase the damping (kd)
                 ".*_shoulder_.*_joint": 3.0,
                 ".*_elbow_joint": 2.5,
                 ".*_wrist_.*_joint": 2.5,
             },
            armature=None,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                "left_hand_Joint1_1",
                "left_hand_Joint2_1",
                "right_hand_Joint1_1",
                "right_hand_Joint2_1",
            ],
            effort_limit=None,  # increase the torque limit
            velocity_limit=None,  # set the velocity limit to 0
            stiffness=800.0,    # increase the stiffness (kp)
            damping=3.0,        # increase the damping (kd)
            friction=200.0,
            armature=None,
        ),

    },
)

