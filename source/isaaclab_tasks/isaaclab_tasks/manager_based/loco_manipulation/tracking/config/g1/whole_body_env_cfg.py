# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Whole body control environment configuration for Unitree G1 humanoid robot.

This environment configuration implements a flexible whole body control system where:
- 4 joint groups: Hand, Arm, Waist, Leg
- 3 policy types per group: RL, IL, IK
- Upper Body = Hand + Arm
- Lower Body = Waist + Leg
- Configurable policy assignment per group
"""

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
from . import mdp as g1_mdp

from .robots.unitree import G129_CFG_WITH_DEX3_BASE_FLOATING

# G129 joint names organized by groups
HAND_JOINT_NAMES = [
    ".*_hand_index_0_joint",
    ".*_hand_middle_0_joint",
    ".*_hand_thumb_0_joint",
    ".*_hand_index_1_joint",
    ".*_hand_middle_1_joint",
    ".*_hand_thumb_1_joint",
    ".*_hand_thumb_2_joint",
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

WAIST_JOINT_NAMES = [
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
]

LEG_JOINT_NAMES = [
    ".*_hip_yaw_joint",
    ".*_hip_roll_joint", 
    ".*_hip_pitch_joint",
    ".*_knee_joint",
    ".*_ankle_pitch_joint",
    ".*_ankle_roll_joint",
]

# Combined groups for convenience
UPPER_BODY_JOINTS = HAND_JOINT_NAMES + ARM_JOINT_NAMES
LOWER_BODY_JOINTS = WAIST_JOINT_NAMES + LEG_JOINT_NAMES
ALL_JOINTS = HAND_JOINT_NAMES + ARM_JOINT_NAMES + WAIST_JOINT_NAMES + LEG_JOINT_NAMES

# Link names for contact sensing and rewards
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
class G1WholeBodySceneCfg(InteractiveSceneCfg):
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
    robot: ArticulationCfg = copy.deepcopy(G129_CFG_WITH_DEX3_BASE_FLOATING)
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
class G1WholeBodyRewardsCfg:
    """Reward terms for the G1 whole body control task."""

    # Velocity tracking rewards
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )

    # Termination penalty
    termination_penalty = RewTerm(
        func=mdp.is_terminated,
        weight=-100.0,
    )

    '''
    # Undesired contacts penalty
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=ARM_LINK_NAMES + HAND_LINK_NAMES), "threshold": 1.0},
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

    joint_vel_hip_yaw = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_hip_yaw_joint"])},
    )

    flat_orientation_l2 = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-1.0,
    )

    # Walking rewards
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.125,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"]),
            "command_name": "base_velocity",
            "threshold": 0.8,
        },
    )

    # Joint regularization rewards (for RL-controlled joints)
    joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS)},
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
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS)},
    )

    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS),
            "soft_ratio": 0.9,
        },
    )
    '''

@configclass
class G1WholeBodyActionsCfg:
    """Action specifications for the MDP with configurable policy types."""

    joint_pos = g1_mdp.WholeBodyJointPositionActionCfg(
        asset_name="robot",
        hand_joint_names=HAND_JOINT_NAMES,
        arm_joint_names=ARM_JOINT_NAMES,
        waist_joint_names=WAIST_JOINT_NAMES,
        leg_joint_names=LEG_JOINT_NAMES,
        # Policy configuration - modify these as needed
        hand_policy=g1_mdp.PolicyType.RL,    # IK, IL, or RL
        arm_policy=g1_mdp.PolicyType.RL,     # IK, IL, or RL  
        waist_policy=g1_mdp.PolicyType.RL,   # IK, IL, or RL
        leg_policy=g1_mdp.PolicyType.RL,     # IK, IL, or RL
        scale=0.5,
        use_default_offset=True,
        # Pink IK configuration (optional - set paths if available)
        urdf_path=None,  # Set to G1 URDF path to enable Pink IK
        mesh_path=None,  # Set to G1 mesh path if needed
        # Trajectory generator configuration (for IK policy)
        trajectory_generator_type=None,  # Options: 'circular', 'linear', 'custom'
        trajectory_generator_params=None,  # Custom parameters for trajectory generator
        # Upper body IL policy configuration
        upper_body_policy_type=None,  # Options: 'separate', 'unified'
        upper_body_policy_model_path=None,  # Path to IL model(s)
    )


@configclass
class G1WholeBodyObservationsCfg:
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
        # Joint states for all joints (for awareness)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=3
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS)},
            noise=Unoise(n_min=-1.0, n_max=1.0),
            history_length=2
        )
        # Action history (only for RL-controlled joints)
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

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "base_velocity"},
        )
        # Joint states with history for all joints
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS)},
            history_length=3
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS)},
            history_length=2
        )
        
        # Action history (only for RL-controlled joints)
        actions = ObsTerm(func=mdp.last_action, history_length=1)

        def __post_init__(self) -> None:
            self.enable_corruption = False  # Critic gets clean observations
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1WholeBodyCommandsCfg:
    """Command generators for base velocity control."""
    
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.25,
        rel_heading_envs=1.0,
        heading_command=True,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class G1WholeBodyEventsCfg:
    """Disturbance and domain-randomization events."""

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
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    # randomize the center of mass of the base to simulate a payload being carried.
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # randomize the joint parameters of the robot (all joints).
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=ALL_JOINTS),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )


@configclass
class G1WholeBodyTerminationsCfg:
    """Termination terms for the MDP."""

    time_out = TermTerm(func=mdp.time_out, time_out=True)
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
            "threshold": 1.0,
        },
    )
    '''
    base_orientation = TermTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 1.0},
    )
    '''
    base_height = TermTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.3},
    )


@configclass
class G1WholeBodyCurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class G1WholeBodyEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for G1 whole body control."""
    
    # Scene
    scene: G1WholeBodySceneCfg = G1WholeBodySceneCfg(num_envs=4096, env_spacing=2.5)
    # Core configurations
    rewards: G1WholeBodyRewardsCfg = G1WholeBodyRewardsCfg()
    actions: G1WholeBodyActionsCfg = G1WholeBodyActionsCfg()
    observations: G1WholeBodyObservationsCfg = G1WholeBodyObservationsCfg()
    commands: G1WholeBodyCommandsCfg = G1WholeBodyCommandsCfg()
    # Extra settings
    terminations: G1WholeBodyTerminationsCfg = G1WholeBodyTerminationsCfg()
    events: G1WholeBodyEventsCfg = G1WholeBodyEventsCfg()
    curriculum: G1WholeBodyCurriculumCfg = G1WholeBodyCurriculumCfg()

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


class G1WholeBodyEnvCfg_PLAY(G1WholeBodyEnvCfg):
    """Smaller play configuration for interactive testing."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Make a smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable randomization for play.
        self.observations.policy.enable_corruption = False


# Specialized configurations for different control scenarios

@configclass  
class G1WholeBodyEnvCfg_UpperBodyIK(G1WholeBodyEnvCfg):
    """Configuration where upper body uses IK and lower body uses RL."""
    
    def __post_init__(self) -> None:
        super().__post_init__()
        # Upper body uses IK
        self.actions.joint_pos.hand_policy = g1_mdp.PolicyType.IK
        self.actions.joint_pos.arm_policy = g1_mdp.PolicyType.IK
        # Lower body uses RL
        self.actions.joint_pos.waist_policy = g1_mdp.PolicyType.RL
        self.actions.joint_pos.leg_policy = g1_mdp.PolicyType.RL
        # IK configuration
        self.actions.joint_pos.urdf_path = "/home/eric/sequor_robotics/sequor_sim/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/loco_manipulation/tracking/config/g1/robots/g1_29dof_with_hand.urdf"
        self.actions.joint_pos.mesh_path = None
        # Trajectory generator configuration for IK
        self.actions.joint_pos.trajectory_generator_type = "circular"
        self.actions.joint_pos.trajectory_generator_params = {
            "radius": 0.1,
            "frequency": 0.5
        }


@configclass
class G1WholeBodyEnvCfg_UpperBodyIL(G1WholeBodyEnvCfg):
    """Configuration where upper body uses IL and lower body uses RL."""
    
    def __post_init__(self) -> None:
        super().__post_init__()
        # Upper body uses IL
        self.actions.joint_pos.hand_policy = g1_mdp.PolicyType.IL
        self.actions.joint_pos.arm_policy = g1_mdp.PolicyType.IL
        # Lower body uses RL
        self.actions.joint_pos.waist_policy = g1_mdp.PolicyType.RL
        self.actions.joint_pos.leg_policy = g1_mdp.PolicyType.RL
        # IL configuration - unified mode example
        self.actions.joint_pos.upper_body_policy_type = "unified"
        # self.actions.joint_pos.upper_body_policy_model_path = "/path/to/unified_il_model.pt"  # Uncomment to use real model


@configclass
class G1WholeBodyEnvCfg_FullRL(G1WholeBodyEnvCfg):
    """Configuration where all joints are RL-controlled."""
    
    def __post_init__(self) -> None:
        super().__post_init__()
        # All joints use RL
        self.actions.joint_pos.hand_policy = g1_mdp.PolicyType.RL
        self.actions.joint_pos.arm_policy = g1_mdp.PolicyType.RL
        self.actions.joint_pos.waist_policy = g1_mdp.PolicyType.RL
        self.actions.joint_pos.leg_policy = g1_mdp.PolicyType.RL


# Play configurations for interactive testing

@configclass
class G1WholeBodyEnvCfg_UpperBodyIK_PLAY(G1WholeBodyEnvCfg_UpperBodyIK):
    """Play configuration for UpperBodyIK environment."""
    
    def __post_init__(self) -> None:
        super().__post_init__()
        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class G1WholeBodyEnvCfg_UpperBodyIL_PLAY(G1WholeBodyEnvCfg_UpperBodyIL):
    """Play configuration for UpperBodyIL environment."""
    
    def __post_init__(self) -> None:
        super().__post_init__()
        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable randomization for play
        self.observations.policy.enable_corruption = False


@configclass
class G1WholeBodyEnvCfg_FullRL_PLAY(G1WholeBodyEnvCfg_FullRL):
    """Play configuration for FullRL environment."""
    
    def __post_init__(self) -> None:
        super().__post_init__()
        # Make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable randomization for play
        self.observations.policy.enable_corruption = False
