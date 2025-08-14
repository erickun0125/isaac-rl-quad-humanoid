# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from .robots.unitree import G129_CFG_WITH_DEX3_BASE_FIX, G129_CFG_WITH_DEX3_FLOATING

from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as TermTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as manipulation_mdp
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg, EventCfg

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
ALL_CONTROLLED_JOINTS = LEG_JOINT_NAMES + WAIST_JOINT_NAMES + ARM_JOINT_NAMES + HAND_JOINT_NAMES


@configclass
class G1LocoManipRewards:
    joint_vel_hip_yaw = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES + HAND_JOINT_NAMES)},
    )

    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.5,
        params={"target_height": 0.7},
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "command_name": "base_velocity",
            "threshold": 0.8,
        },
    )

    undesired_contacts = None

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-5.0,
    )

    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINTS)},
    )

    joint_accel_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + WAIST_JOINT_NAMES)},
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )

    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINTS)},
    )

    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINTS),
            "soft_ratio": 0.9,
        },
    )

    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-0.5,
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.05,
    )

    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES + WAIST_JOINT_NAMES)},
    )

    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

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

    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-100.0,
    )


@configclass
class G1LocoManipObservations:

    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.1, n_max=0.1),
        )
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
        left_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "left_ee_pose"},
        )
        right_ee_pose_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "right_ee_pose"},
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINTS)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy = PolicyCfg()


@configclass
class G1LocoManipCommands:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.25,
        rel_heading_envs=1.0,
        heading_command=True,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )

    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="left_wrist_yaw_link",  # Updated for G129 DEX3 configuration
        resampling_time_range=(1.0, 3.0),
        debug_vis=True,
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
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.50),
            pos_y=(-0.50, -0.05),
            pos_z=(-0.20, 0.20),
            roll=(-0.1, 0.1),
            pitch=(-0.1, 0.1),
            yaw=(-math.pi / 2.0 - 0.1, -math.pi / 2.0 + 0.1),
        ),
    )


@configclass
class G1Events(EventCfg):
    # Add an external force to simulate a payload being carried.
    left_hand_force = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-1.0, 1.0),
        },
    )

    right_hand_force = EventTermCfg(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-1.0, 1.0),
        },
    )


@configclass
class G1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=LEG_JOINT_NAMES + WAIST_JOINT_NAMES + ARM_JOINT_NAMES + HAND_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )
'''
class G1ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)
'''

@configclass
class G1TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TermTerm(func=mdp.time_out, time_out=True)
    base_contact = TermTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["pelvis"]),
            "threshold": 1.0,
        },
    )
    base_orientation = TermTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 2.0},
    )


@configclass
class G1LocoManipEnvCfg(LocomotionVelocityRoughEnvCfg):
    rewards: G1LocoManipRewards = G1LocoManipRewards()
    observations: G1LocoManipObservations = G1LocoManipObservations()
    commands: G1LocoManipCommands = G1LocoManipCommands()
    actions: G1ActionsCfg = G1ActionsCfg()
    terminations: G1TerminationsCfg = G1TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 14.0

        # Replace robot with G129 (with DEX3 hands and floating base)
        import copy
        g1_robot_cfg = copy.deepcopy(G129_CFG_WITH_DEX3_FLOATING)
        g1_robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
        self.scene.robot = g1_robot_cfg

        # Update events for G1 body names
        if hasattr(self.events, 'add_base_mass') and self.events.add_base_mass is not None:
            self.events.add_base_mass.params["asset_cfg"] = SceneEntityCfg("robot", body_names="pelvis")
        if hasattr(self.events, 'base_external_force_torque') and self.events.base_external_force_torque is not None:
            self.events.base_external_force_torque.params["asset_cfg"] = SceneEntityCfg("robot", body_names="pelvis")
        if hasattr(self.events, 'base_com') and self.events.base_com is not None:
            self.events.base_com.params["asset_cfg"] = SceneEntityCfg("robot", body_names="pelvis")
        
        # Rewards:
        self.rewards.flat_orientation_l2.weight = -10.5
        self.rewards.termination_penalty.weight = -100.0

        # Change terrain to flat.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # Remove height scanner and terrain curriculum if they exist
        if hasattr(self.scene, 'height_scanner'):
            self.scene.height_scanner = None
        if hasattr(self.observations.policy, 'height_scan'):
            self.observations.policy.height_scan = None
        if hasattr(self.curriculum, 'terrain_levels'):
            self.curriculum.terrain_levels = None


class G1LocoManipEnvCfg_PLAY(G1LocoManipEnvCfg):

    def __post_init__(self) -> None:
        super().__post_init__()

        # Make a smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable randomization for play.
        self.observations.policy.enable_corruption = False
        # Remove random pushing if it exists
        if hasattr(self.events, 'base_external_force_torque'):
            self.events.base_external_force_torque = None
        if hasattr(self.events, 'push_robot'):
            self.events.push_robot = None