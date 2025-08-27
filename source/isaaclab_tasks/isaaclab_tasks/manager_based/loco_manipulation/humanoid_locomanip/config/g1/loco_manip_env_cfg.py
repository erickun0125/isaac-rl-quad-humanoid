# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import copy
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as TermTerm,
    CurriculumTermCfg as CurrTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as manipulation_mdp
from .mdp import locomanip_mdp

from .robots.unitree import G129_CFG_WITH_DEX3_BASE_FLOATING_FOR_LOCOMANIP

# G129 specific joint names (29 DOF with DEX3 hands)
LEG_JOINT_NAMES = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint", 
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]

WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

ARM_JOINT_NAMES = [
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint", 
    ".*_shoulder_yaw_joint",
    ".*_elbow_joint",
    ".*_wrist_roll_joint",
    ".*_wrist_pitch_joint",
    ".*_wrist_yaw_joint",
]

HAND_JOINT_NAMES = [
    ".*_hand_index_0_joint",
    ".*_hand_middle_0_joint",
    ".*_hand_thumb_0_joint",
    ".*_hand_index_1_joint",
    ".*_hand_middle_1_joint",
    ".*_hand_thumb_1_joint",
    ".*_hand_thumb_2_joint",
]

# Combined joint names for easier reference
# without HAND_JOINT_NAMES
CONTROLLED_JOINTS = LEG_JOINT_NAMES + WAIST_JOINT_NAMES + ARM_JOINT_NAMES

# Link names grouped by body region
LEG_LINK_NAMES = [
    ".*_hip_pitch_link",
    ".*_hip_roll_link",
    ".*_hip_yaw_link",
    ".*_knee_link",
    ".*_ankle_pitch_link",
    ".*_ankle_roll_link",
]

LOWER_BASE_LINK_NAMES = [
    "pelvis",
    "imu_in_pelvis",
    "pelvis_contour_link",
    "waist_yaw_link",
    "waist_roll_link",
]

UPPER_BASE_LINK_NAMES = [
    "torso_link",
    "imu_in_torso",
    "head_link",
    "d435_link",
    "mid360_link",
    "logo_link",
]

ARM_LINK_NAMES = [
    ".*_shoulder_pitch_link",
    ".*_shoulder_roll_link",
    ".*_shoulder_yaw_link",
    ".*_elbow_link",
    ".*_wrist_roll_link",
    ".*_wrist_pitch_link",
    ".*_wrist_yaw_link",
]

HAND_LINK_NAMES = [
    ".*_hand_camera_base_link",
    ".*_hand_palm_link",
    ".*_hand_index_0_link",
    ".*_hand_index_1_link",
    ".*_hand_middle_0_link",
    ".*_hand_middle_1_link",
    ".*_hand_thumb_0_link",
    ".*_hand_thumb_1_link",
    ".*_hand_thumb_2_link",
]


@configclass
class G1LocoManipSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # Ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # Robots
    robot: ArticulationCfg = copy.deepcopy(G129_CFG_WITH_DEX3_BASE_FLOATING_FOR_LOCOMANIP)
    robot.prim_path = "{ENV_REGEX_NS}/Robot"
    # Sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # Lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class G1LocoManipRewardsCfg:
    """Reward terms for the G1 loco-manipulation task."""

    # Velocity tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.25,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # Termination & Undesired Contacts penalty
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-100.0,
    )

    '''
    # End-effector tracking rewards
    left_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            "command_name": "left_ee_pose",
        },
    )

    left_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            "std": 0.05,
            "command_name": "left_ee_pose",
        },
    )

    left_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            "command_name": "left_ee_pose",
        },
    )

    right_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            "command_name": "right_ee_pose",
        },
    )

    right_ee_pos_tracking_fine_grained = RewTerm(
        func=manipulation_mdp.position_command_error_tanh,
        weight=2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            "std": 0.05,
            "command_name": "right_ee_pose",
        },
    )

    right_end_effector_orientation_tracking = RewTerm(
        func=manipulation_mdp.orientation_command_error,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            "command_name": "right_ee_pose",
        },
    )

    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=ARM_LINK_NAMES), "threshold": 1.0},
    )
    '''

    # Pose rewards
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=ARM_JOINT_NAMES,
            )
        },
    )    

    joint_deviation_shoulder_roll = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[".*_shoulder_roll_joint"],
            )
        },
    )        

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINT_NAMES)},
    )

    # Stability rewards
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.5,
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.1,
    )
    # torso_link is the root link, so it is same as the base orientation
    torso_orientation_l2 = RewTerm(
        func=locomanip_mdp.torso_orientation_l2,
        weight=-0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )

    torso_backward_tilt_penalty = RewTerm(
        func=locomanip_mdp.torso_backward_tilt_penalty,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )

    # Walking rewards
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
            "threshold": 0.1,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    foot_clearance = RewTerm(
        func=locomanip_mdp.foot_clearance_reward,
        weight=0.25,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "target_height": 0.2,
            "std": 0.05,
            "tanh_mult": 8.0,
        },
    )    

    # Joint regularization rewards
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS)},
    )

    joint_accel_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS)},
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.005,
    )


    '''
    #limit penalties
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS)},
    )

    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS),
            "soft_ratio": 0.9,
        },
    )
    '''


@configclass
class G1LocoManipActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names= CONTROLLED_JOINTS,
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class G1LocoManipObservationsCfg:
    """Observation terms grouped for policy and critic networks."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations used as the actor input."""

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        '''
        left_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_ee_pose"},
        )
        right_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_ee_pose"},
        )
        '''
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=3
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            history_length=2
        )
        actions = ObsTerm(func=mdp.last_action, history_length=1)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations - includes privileged information."""

        # Privileged base state information
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_pos_z = ObsTerm(func=mdp.base_pos_z)
        foot_contact = ObsTerm(
            func=locomanip_mdp.foot_contact,
            params={
                "sensor_name": "contact_forces",
                "threshold": 1.0,
            },
        )

        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
        )
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        '''
        left_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_ee_pose"},
        )
        right_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_ee_pose"},
        )
        '''
        # Joint states with history
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)},
            history_length=3
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)},
            history_length=2
        )
        
        # Action history
        actions = ObsTerm(func=mdp.last_action, history_length=1)

        def __post_init__(self) -> None:
            self.enable_corruption = False  # Critic gets clean observations
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1LocoManipCommandsCfg:
    """Command generators for base velocity and end-effector poses."""
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.2,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.5, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )
    '''
    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="left_wrist_yaw_link",  # Updated for G129 DEX3 configuration
        resampling_time_range=(1.0, 3.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.50),
            pos_y=(0.05, 0.50),
            pos_z=(-0.20, 0.20),
            roll=(-0.1, 0.1),
            pitch=(-0.1, 0.1),
            yaw=(math.pi / 2.0 - 0.1, math.pi / 2.0 + 0.1),
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="right_wrist_yaw_link",  # Updated for G129 DEX3 configuration
        resampling_time_range=(1.0, 3.0),
        debug_vis=False,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.50),
            pos_y=(-0.50, -0.05),
            pos_z=(-0.20, 0.20),
            roll=(-0.1, 0.1),
            pitch=(-0.1, 0.1),
            yaw=(-math.pi / 2.0 - 0.1, -math.pi / 2.0 + 0.1),
        ),
    )
    '''

@configclass
class G1LocoManipEventsCfg:
    """Disturbance and domain-randomization events."""
    
    reset_scene_to_default = EventTerm(
        func=mdp.reset_scene_to_default,
        mode="reset",
        params={
            "reset_joint_targets": True,
        },
    )
    
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )
    
    # Add an external force to simulate a payload being carried.

    '''
    right_hand_force = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            "force_range": (-1.0, 1.0),
            "torque_range": (-1.0, 1.0),
        },
    )

    left_hand_force = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            "force_range": (-1.0, 1.0),
            "torque_range": (-1.0, 1.0),
        },
    )
    

    torso_wrench = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "force_range": (-1.0, 1.0),
            "torque_range": (-1.0, 1.0),
        },
    )
    '''
    # push the base to simulate a pushing force
    push_base = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 15.0),
        params={
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )
    
    # randomize the physics material of the robot.
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.6, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    # add mass to the base to simulate a payload being carried.
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # randomize the center of mass of the base to simulate a payload being carried.
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # randomize the joint parameters of the robot.
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )

@configclass
class G1LocoManipTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TermTerm(func=mdp.time_out, time_out=True)

    base_height = TermTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.52},
    )    
    
    base_orientation = TermTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.78},
    )

    '''
    base_contact_pelvis = TermTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis"]),
            "threshold": 1.0,
        },
    )
    
    base_contact_torso = TermTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["torso_link"]),
            "threshold": 10.0,
        },
    )
    '''

@configclass
class G1LocoManipCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    # Walking reward weight curriculum - increase the weight of the walking reward
    walking_reward_curriculum = CurrTerm(
        func=locomanip_mdp.reward_weight_curriculum,
        params={
            "reward_term_names": ["feet_air_time", "feet_slide", "foot_clearance"],
            "tracking_reward_name": "track_lin_vel_xy_exp",
            "min_ratio": 0.1,
            "max_ratio": 1.0,
            "reward_threshold": 0.75,
            "ratio_step": 0.1,
        }
    )    

    # Stability reward weight curriculum - increase the weight of the stability reward
    stability_reward_curriculum = CurrTerm(
        func=locomanip_mdp.reward_weight_curriculum,
        params={
            "reward_term_names": ["lin_vel_z_l2", "ang_vel_xy_l2"],
            "tracking_reward_name": "track_lin_vel_xy_exp",
            "min_ratio": 0.1,
            "max_ratio": 1.0,
            "reward_threshold": 0.75,
            "ratio_step": 0.05,
        }
    )
    
    # Joint deviation weight curriculum - gradually decrease penalty weights
    joint_deviation_curriculum = CurrTerm(
        func=locomanip_mdp.reward_weight_curriculum,
        params={
            "reward_term_names": ["joint_deviation_arms", "joint_deviation_hip"],
            "tracking_reward_name": "track_lin_vel_xy_exp",
            "min_ratio": 0.25,
            "max_ratio": 1.0,
            "reward_threshold": 0.8,
            "ratio_step": -0.01,
        }
    )    
    '''
    # Velocity command curriculum - gradually increase velocity ranges
    velocity_curriculum = CurrTerm(
        func=locomanip_mdp.velocity_range_curriculum,
        params={
            "lin_reward_term": "track_lin_vel_xy_exp",
            "ang_reward_term": "track_ang_vel_z_exp",
            "lin_vel_x_limit": (-1.0, 1.5),
            "lin_vel_y_limit": (-1.0, 1.0),
            "ang_vel_z_limit": (-1.0, 1.0),
            "reward_threshold": 0.85,
            "delta_step": 0.1,
        }
    )

    # Disturbance curriculum - gradually increase both push force and external force intensity based on termination rate and reward performance
    disturbance_curriculum = CurrTerm(
        func=locomanip_mdp.disturbance_range_curriculum,
        params={
            "termination_term_names": ["base_height", "base_orientation"],
            "reward_term_name": "track_lin_vel_xy_exp",  # Monitor velocity tracking performance
            "push_event_names": ["push_base"],
            "external_force_event_names": ["torso_wrench"],
            "max_push_velocity": 0.5,
            "max_force_magnitude": 5.0,
            "max_torque_magnitude": 2.0,
            "termination_threshold": 0.1,  # Increase disturbance if termination rate < 10%
            "reward_threshold": 0.8,  # Increase disturbance if reward performance > 80% of weight
            "push_increase_step": 0.05,
            "force_increase_step": 0.2,
        }
    )
    '''
@configclass
class G1LocoManipEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for G1 loco-manipulation."""
    # Scene
    scene: G1LocoManipSceneCfg = G1LocoManipSceneCfg(num_envs=4096, env_spacing=2.5)
    # r,a,o,c
    rewards: G1LocoManipRewardsCfg = G1LocoManipRewardsCfg()
    actions: G1LocoManipActionsCfg = G1LocoManipActionsCfg()
    observations: G1LocoManipObservationsCfg = G1LocoManipObservationsCfg()
    commands: G1LocoManipCommandsCfg = G1LocoManipCommandsCfg()
    # Extra settings
    terminations: G1LocoManipTerminationsCfg = G1LocoManipTerminationsCfg()
    events: G1LocoManipEventsCfg = G1LocoManipEventsCfg()
    curriculum: G1LocoManipCurriculumCfg = G1LocoManipCurriculumCfg()

    def __post_init__(self) -> None:
        # General settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # Simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        # Flat terrain has no generator; disable terrain curriculum if present
        if hasattr(self.curriculum, "terrain_levels"):
            self.curriculum.terrain_levels = None

class G1LocoManipEnvCfg_PLAY(G1LocoManipEnvCfg):
    """Smaller play configuration for interactive testing."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Make a smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable randomization for play.
        self.observations.policy.enable_corruption = False
        # Remove disturbance events for play mode
        if hasattr(self.events, "left_hand_force"):
            self.events.left_hand_force = None
        if hasattr(self.events, "right_hand_force"):
            self.events.right_hand_force = None
        if hasattr(self.events, "torso_wrench"):
            self.events.torso_wrench = None
        if hasattr(self.events, "push_base"):
            self.events.push_base = None