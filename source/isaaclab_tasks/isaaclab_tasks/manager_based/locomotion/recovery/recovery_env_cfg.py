# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.recovery.mdp as mdp

##
# Pre-defined configs
##
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

##
# Pre-defined robot configs
##
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot for recovery training."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=3,  # Start with easier terrains for recovery
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
    # robots - Go2 as default
    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # sensors
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    '''
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25, use_default_offset=False,
                                                offset={
                                                    ".*L_hip_joint": 0.4,      # 모든 Left hip
                                                    ".*R_hip_joint": -0.4,     # 모든 Right hip  
                                                    "F[LR]_thigh_joint": 1.36,  # Front thigh (FL, FR)
                                                    "R[LR]_thigh_joint": 1.36,  # Rear thigh (RL, RR)
                                                    ".*_calf_joint": -2.35,     # 모든 calf joints
                                                } 
                                           )  # Go2 optimized scale
    '''
    joint_pos = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25,)

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1), history_length=2)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            history_length=2
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=2)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=2)
        actions = ObsTerm(func=mdp.last_action, history_length=2)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1), history_length=2)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            history_length=2
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=2)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=2)
        actions = ObsTerm(func=mdp.last_action, history_length=2)
        progress_ratio = ObsTerm(func=mdp.episode_progress)  # Normalized progress (0-1)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True


    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()

@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),  # Go2 optimized mass range
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.005, 0.005)},  # Go2 optimized COM range
        },
    )

    # reset - use physics simulation reset for realistic settling (Go2 optimized)
    reset_with_physics_simulation = EventTerm(
        func=mdp.reset_with_physics_simulation,
        mode="reset",
        params={
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-3.14, 3.14)},
            "roll_range": (-1.5, 1.5),  # Allow more extreme orientations for realistic falling
            "pitch_range": (-1.5, 1.5),  # Wider pitch range for diverse fallen states
            "height_range": (0.35, 0.55),  # Drop from higher for realistic physics simulation
            "simulation_time": 0.5,  # 1 second of physics simulation before policy starts
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- main recovery task rewards 
    '''
    target_configuration = RewTerm(
        func=mdp.target_configuration,
        weight=1.0,
        params={
            "flat_orientation_weight": -1.0,    # Weight for upright orientation reward
            "joint_pose_weight": 1.0,          # Weight for joint pose reward  
            "final_config_weight": 1.0,        # Weight for final configuration bonus
            "sub_orientation_threshold": 0.524,  # 30 degrees - threshold for joint pose reward
            "final_orientation_threshold": 0.087,  # 5 degrees - threshold for final configuration
            "final_joint_threshold": 0.05,      # Max joint error for final configuration (radians)
            "std": 1.0,                        # Standard deviation for joint pose reward normalization
            "big_reward": 100.0,                # Bonus reward for achieving final configuration
        },
    )
    '''
    # -- separate target configuration rewards: flat_orientation, joint_pose, final_configuration
    flat_orientation = RewTerm(
        func=mdp.flat_orientation,
        weight=1.5,  # Positive weight (function already applies negative internally)
    )
    joint_pose = RewTerm(
        func=mdp.joint_pose,
        weight=1.0,  # Positive weight for joint pose reward
    )
    foot_contact = RewTerm(
        func=mdp.foot_contact,
        weight=0.1,  # Positive weight for foot contact reward
    )
    final_configuration = RewTerm(  
        func=mdp.final_configuration,
        weight=10.0,  # Positive weight for final configuration bonus
    )



    # -- penalty terms
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_threshold_penalty, weight=-0.1, params={"threshold": 0.5})
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_threshold_penalty, weight=-0.06, params={"threshold":2.0})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)

    
    # -- contact penalties (Go2 specific)
    '''
    contact_force_penalty = RewTerm(
        func=mdp.contact_forces,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*thigh", ".*calf"]), "threshold": 50.0},  # Go2 body naming
    )
    '''
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # Use simplified curriculum for Go2 recovery
    pass  # No complex terrain curriculum needed for pure recovery task


##
# Environment configuration
##


@configclass
class LocomotionRecoveryRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for Go2 locomotion recovery environment."""

    # Scene settings - Go2 optimized with reduced GPU usage
    scene: MySceneCfg = MySceneCfg(num_envs=512, env_spacing=2.5)  # Significantly reduced for GPU memory
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings - Go2 optimized
        self.decimation = 4
        self.episode_length_s = 5.0  # Optimal for Go2 recovery learning
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        '''        
        # Reduce GPU memory usage for PhysX
        self.sim.physx.gpu_max_rigid_contact_count = 2**19  # Reduced from default
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**18  # Reduced capacity
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**16
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**16
        self.sim.physx.friction_correlation_distance = 0.025
        self.sim.physx.friction_offset_threshold = 0.04
        self.sim.physx.bounce_threshold_velocity = 0.2
        '''
        # Go2 specific sensor settings
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Go2 optimized terrain settings
        if self.scene.terrain.terrain_generator is not None:
            # Scale terrain features for Go2 size
            self.scene.terrain.terrain_generator.curriculum = False  # No curriculum for pure recovery
            # Set terrain difficulty appropriate for Go2
            self.scene.terrain.max_init_terrain_level = 2
