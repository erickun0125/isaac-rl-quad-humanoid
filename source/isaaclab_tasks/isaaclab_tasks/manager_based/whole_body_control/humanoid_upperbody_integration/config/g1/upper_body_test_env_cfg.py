# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Upper body test environment configuration for Unitree G1 humanoid robot.

This environment configuration is designed specifically for testing upper body control:
- Focuses on Hand and Arm joints (upper body)
- Waist and Leg joints are fixed/disabled
- Uses G129_CFG_WITH_DEX3_BASE_FIX robot configuration
- Suitable for upper body manipulation tasks and testing
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

from .robots.unitree_fixed import G129_CFG_WITH_DEX3_BASE_FIX

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
class G1UpperBodyTestSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot for upper body testing."""

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
    robot: ArticulationCfg = copy.deepcopy(G129_CFG_WITH_DEX3_BASE_FIX)
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
class G1UpperBodyTestRewardsCfg:
    """Reward terms for the G1 upper body test task."""


@configclass
class G1UpperBodyTestActionsCfg:
    """Action specifications for the MDP focused on upper body control."""

    joint_pos = g1_mdp.WholeBodyJointPositionActionCfg(
        asset_name="robot",
        hand_joint_names=HAND_JOINT_NAMES,
        arm_joint_names=ARM_JOINT_NAMES,
        waist_joint_names=WAIST_JOINT_NAMES,
        leg_joint_names=LEG_JOINT_NAMES,
        # Policy configuration - upper body IK + lower body RL
        hand_policy=g1_mdp.PolicyType.IK,     # IK for upper body
        arm_policy=g1_mdp.PolicyType.IK,      # IK for upper body
        waist_policy=g1_mdp.PolicyType.RL,    # RL for lower body
        leg_policy=g1_mdp.PolicyType.RL,      # RL for lower body
        scale=0.0,
        use_default_offset=True,
        # Trajectory generator configuration (for IK policy)
        trajectory_generator_type="circular",  # Circular trajectory for upper body
        trajectory_generator_params={
            "center": (0.5, 0.0, 0.5),
            "radius": 0.3,
            "frequency": 0.01
        },
        # Upper body IL policy configuration
        upper_body_policy_type=None,  # Options: 'separate', 'unified'
        upper_body_policy_model_path=None,  # Path to IL model(s)
    )


@configclass
class G1UpperBodyTestObservationsCfg:
    """Observation terms grouped for policy and critic networks."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Policy observations used as the actor input."""

        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        def __post_init__(self) -> None:
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations - includes privileged information."""

        projected_gravity = ObsTerm(func=mdp.projected_gravity)


        def __post_init__(self) -> None:
            self.enable_corruption = False  # Critic gets clean observations
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class G1UpperBodyTestCommandsCfg:
    """Command generators for upper body testing."""
    
    # No velocity commands needed for fixed base testing
    pass


@configclass
class G1UpperBodyTestEventsCfg:
    """Disturbance and domain-randomization events for upper body testing."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )

    # randomize the joint parameters of the robot (upper body joints only).
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=UPPER_BODY_JOINTS),
            "stiffness_distribution_params": (0.95, 1.05),
            "damping_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )


@configclass
class G1UpperBodyTestTerminationsCfg:
    """Termination terms for the upper body test MDP."""

    time_out = TermTerm(func=mdp.time_out, time_out=True)


@configclass
class G1UpperBodyTestCurriculumCfg:
    """Curriculum terms for the upper body test MDP."""

    # No terrain curriculum needed for fixed base testing
    pass


@configclass
class G1UpperBodyTestEnvCfg(ManagerBasedRLEnvCfg):
    """Environment configuration for G1 upper body testing."""
    
    # Scene
    scene: G1UpperBodyTestSceneCfg = G1UpperBodyTestSceneCfg(num_envs=1024, env_spacing=2.5)
    # Core configurations
    rewards: G1UpperBodyTestRewardsCfg = G1UpperBodyTestRewardsCfg()
    actions: G1UpperBodyTestActionsCfg = G1UpperBodyTestActionsCfg()
    observations: G1UpperBodyTestObservationsCfg = G1UpperBodyTestObservationsCfg()
    commands: G1UpperBodyTestCommandsCfg = G1UpperBodyTestCommandsCfg()
    # Extra settings
    terminations: G1UpperBodyTestTerminationsCfg = G1UpperBodyTestTerminationsCfg()
    events: G1UpperBodyTestEventsCfg = G1UpperBodyTestEventsCfg()
    curriculum: G1UpperBodyTestCurriculumCfg = G1UpperBodyTestCurriculumCfg()

    def __post_init__(self) -> None:
        # General settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # Simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


class G1UpperBodyTestEnvCfg_PLAY(G1UpperBodyTestEnvCfg):
    """Smaller play configuration for interactive upper body testing."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Make a smaller scene for play.
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # Disable randomization for play.
        self.observations.policy.enable_corruption = False
