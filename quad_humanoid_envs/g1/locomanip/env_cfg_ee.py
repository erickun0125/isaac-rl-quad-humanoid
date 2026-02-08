"""End-effector tracking environment configuration for G1 humanoid loco-manipulation.

This configuration extends the base loco-manipulation environment with:
- End-effector pose commands (left/right wrist)
- EE tracking rewards
- Alive reward with curriculum
- Narrower velocity command ranges
- Reduced domain randomization (reset events only)
"""

import math

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
import isaaclab_tasks.manager_based.manipulation.reach.mdp as manipulation_mdp
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
)
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp as locomanip_mdp
from .env_cfg import (
    ARM_JOINT_NAMES,
    CONTROLLED_JOINTS,
    WAIST_JOINT_NAMES,
    G1LocoManipActionsCfg,
    G1LocoManipEnvCfg,
    G1LocoManipSceneCfg,
    G1LocoManipTerminationsCfg,
)


# ---------------------------------------------------------------------------
# Rewards (extends base with EE tracking and different weights)
# ---------------------------------------------------------------------------

@configclass
class G1LocoManipEERewardsCfg:
    """Reward terms for the G1 loco-manipulation task with end-effector tracking."""

    # Alive reward
    alive_reward = RewTerm(
        func=mdp.is_alive,
        weight=10.0,
    )

    # Velocity tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_yaw_frame_exp,
        weight=1.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_world_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.16)},
    )

    # Termination penalty
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-200.0,
    )

    # End-effector tracking rewards
    left_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-3.0,
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
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="left_wrist_yaw_link"),
            "command_name": "left_ee_pose",
        },
    )

    right_ee_pos_tracking = RewTerm(
        func=manipulation_mdp.position_command_error,
        weight=-3.0,
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
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="right_wrist_yaw_link"),
            "command_name": "right_ee_pose",
        },
    )

    # Pose rewards (different weights from base)
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"])},
    )

    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.01,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINT_NAMES),
        },
    )

    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=WAIST_JOINT_NAMES)},
    )

    torso_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "target_height": 0.72,
        },
    )

    # Stability rewards (different weights from base)
    lin_vel_z_l2 = RewTerm(
        func=mdp.lin_vel_z_l2,
        weight=-1.0,
    )

    ang_vel_xy_l2 = RewTerm(
        func=mdp.ang_vel_xy_l2,
        weight=-0.4,
    )

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.5,
    )

    torso_backward_tilt_penalty = RewTerm(
        func=locomanip_mdp.torso_backward_tilt_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
        },
    )

    # Walking rewards (different weights/params from base)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=4.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )

    foot_clearance = RewTerm(
        func=locomanip_mdp.foot_clearance_reward,
        weight=0.2,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
            "target_height": 0.1,
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

    joint_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-4.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS)},
    )

    action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )


# ---------------------------------------------------------------------------
# Observations (adds EE pose commands)
# ---------------------------------------------------------------------------

@configclass
class G1LocoManipEEObservationsCfg:
    """Observation terms with end-effector pose commands."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations with EE pose commands."""

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
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=3,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-1.0, n_max=1.0),
            history_length=2,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=1)

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations with EE pose commands and privileged information."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_pos_z = ObsTerm(func=mdp.base_pos_z)
        foot_contact = ObsTerm(
            func=locomanip_mdp.foot_contact,
            params={
                "sensor_name": "contact_forces",
                "threshold": 1.0,
            },
        )
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
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
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)},
            history_length=3,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=CONTROLLED_JOINTS, preserve_order=True)},
            history_length=2,
        )
        actions = ObsTerm(func=mdp.last_action, history_length=1)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ---------------------------------------------------------------------------
# Commands (adds EE pose commands, narrower velocity ranges)
# ---------------------------------------------------------------------------

@configclass
class G1LocoManipEECommandsCfg:
    """Command generators for base velocity and end-effector poses."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 5.0),
        rel_standing_envs=0.3,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.25, 0.75),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-0.75, 0.75),
            heading=(-math.pi, math.pi),
        ),
    )

    left_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="left_wrist_yaw_link",
        resampling_time_range=(1.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.30),
            pos_y=(0.0, 0.30),
            pos_z=(-0.10, 0.30),
            roll=(-0.5, 0.5),
            pitch=(-0.5, 0.5),
            yaw=(-0.5, 0.5),
        ),
    )

    right_ee_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name="right_wrist_yaw_link",
        resampling_time_range=(1.0, 5.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(0.10, 0.30),
            pos_y=(-0.30, -0.0),
            pos_z=(-0.10, 0.30),
            roll=(-0.5, 0.5),
            pitch=(-0.5, 0.5),
            yaw=(-0.5, 0.5),
        ),
    )


# ---------------------------------------------------------------------------
# Events (reset-only, no interval disturbances)
# ---------------------------------------------------------------------------

@configclass
class G1LocoManipEEEventsCfg:
    """Events for EE environment -- reset events only, no interval disturbances."""

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


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

@configclass
class G1LocoManipEECurriculumCfg:
    """Curriculum terms for the EE environment."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)

    alive_reward_curriculum = CurrTerm(
        func=locomanip_mdp.reward_weight_curriculum,
        params={
            "reward_term_names": ["alive_reward"],
            "tracking_reward_name": "track_lin_vel_xy_exp",
            "min_ratio": 0.1,
            "max_ratio": 1.0,
            "reward_threshold": 0.5,
            "ratio_step": -0.9,
        },
    )


# ---------------------------------------------------------------------------
# Top-level environment configs
# ---------------------------------------------------------------------------

@configclass
class G1LocoManipEEEnvCfg(G1LocoManipEnvCfg):
    """Environment configuration for G1 loco-manipulation with end-effector tracking.

    Inherits from G1LocoManipEnvCfg and overrides:
    - rewards: adds alive_reward, EE tracking rewards, different weights
    - observations: adds EE pose command observations
    - commands: adds left/right EE pose commands, narrower velocity ranges
    - events: reset-only (no interval disturbances)
    - curriculum: alive_reward curriculum instead of walking/stability curricula
    """

    rewards: G1LocoManipEERewardsCfg = G1LocoManipEERewardsCfg()
    observations: G1LocoManipEEObservationsCfg = G1LocoManipEEObservationsCfg()
    commands: G1LocoManipEECommandsCfg = G1LocoManipEECommandsCfg()
    events: G1LocoManipEEEventsCfg = G1LocoManipEEEventsCfg()
    curriculum: G1LocoManipEECurriculumCfg = G1LocoManipEECurriculumCfg()


class G1LocoManipEEEnvCfg_PLAY(G1LocoManipEEEnvCfg):
    """Smaller play configuration for interactive testing."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
