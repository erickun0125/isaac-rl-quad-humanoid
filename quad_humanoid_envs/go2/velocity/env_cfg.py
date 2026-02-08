"""GO2 velocity environment configurations (flat and rough terrain)."""

from __future__ import annotations

import math

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import UnitreeGo2FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .mdp import (
    modify_physics_material_curriculum,
    modify_external_forces_curriculum,
    modify_push_robot_curriculum,
    modify_velocity_command_curriculum,
    modify_mass_randomization_curriculum,
    modify_reward_weight_curriculum,
    nominal_joint_pos_reward,
    foot_clearance_reward,
    action_smoothness_penalty,
    selective_external_force_torque,
    selective_push_by_setting_velocity,
)


# =============================================================================
# Flat terrain environment
# =============================================================================


@configclass
class FlatObservationsCfg:
    """Asymmetric Actor-Critic observations for flat terrain locomotion."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations - excludes base_lin_vel for better sim-to-real transfer."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=3)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=2)
        actions = ObsTerm(func=mdp.last_action, history_length=2)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations - includes base_lin_vel for better value estimation."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, history_length=3)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, history_length=2)
        actions = ObsTerm(func=mdp.last_action, history_length=2)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class FlatCurriculumCfg:
    """Curriculum configuration for flat terrain progressive training."""

    physics_material = CurrTerm(
        func=modify_physics_material_curriculum,
        params={
            "term_name": "physics_material",
            "num_steps": 22000,
            "warmup_steps": 8000,
            "initial_friction_range": (0.8, 1.6),
            "final_friction_range": (0.8, 2.0),
            "initial_restitution_range": (0.0, 0.0),
            "final_restitution_range": (0.0, 0.1),
        },
    )

    external_forces = CurrTerm(
        func=modify_external_forces_curriculum,
        params={
            "term_name": "base_external_force_torque",
            "num_steps": 22000,
            "warmup_steps": 8000,
            "initial_force_range": (-3.0, 3.0),
            "final_force_range": (-5.0, 5.0),
            "initial_torque_range": (-3.0, 3.0),
            "final_torque_range": (-5.0, 5.0),
        },
    )

    push_robot = CurrTerm(
        func=modify_push_robot_curriculum,
        params={
            "term_name": "push_robot",
            "num_steps": 22000,
            "warmup_steps": 8000,
            "initial_velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)},
            "final_velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            "initial_interval": (10.0, 20.0),
            "final_interval": (6.0, 18.0),
        },
    )

    velocity_command = CurrTerm(
        func=modify_velocity_command_curriculum,
        params={
            "command_name": "base_velocity",
            "num_steps": 22000,
            "warmup_steps": 8000,
            "initial_lin_vel_x": (-0.5, 0.5),
            "final_lin_vel_x": (-1.0, 1.0),
            "initial_lin_vel_y": (-0.5, 0.5),
            "final_lin_vel_y": (-1.0, 1.0),
            "initial_ang_vel_z": (-0.5, 0.5),
            "final_ang_vel_z": (-1.0, 1.0),
        },
    )

    mass_randomization = CurrTerm(
        func=modify_mass_randomization_curriculum,
        params={
            "term_name": "add_base_mass",
            "num_steps": 22000,
            "warmup_steps": 8000,
            "initial_mass_range": (-1.0, 3.0),
            "final_mass_range": (-1.0, 6.0),
        },
    )

    feet_air_time_weight = CurrTerm(
        func=modify_reward_weight_curriculum,
        params={
            "reward_term_name": "feet_air_time",
            "num_steps": 22000,
            "warmup_steps": 8000,
            "initial_weight": 0.25,
            "final_weight": 0.125,
            "decay_type": "cosine",
        },
    )


@configclass
class UnitreeGo2SequorEnvCfg(UnitreeGo2FlatEnvCfg):
    """Enhanced Go2 flat terrain environment with curriculum-based training."""

    observations: FlatObservationsCfg = FlatObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # Physics material event: reset mode for curriculum
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.8, 2.0),
                "dynamic_friction_range": (0.8, 2.0),
                "restitution_range": (0.0, 0.05),
                "num_buckets": 64,
            },
        )

        # PD gain randomization (+/-5%)
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stiffness_distribution_params": (0.95, 1.05),
                "damping_distribution_params": (0.95, 1.05),
                "operation": "scale",
            },
        )

        # Selective external forces (60% stable envs)
        self.events.base_external_force_torque = EventTerm(
            func=selective_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "force_range": (-10.0, 10.0),
                "torque_range": (-5.0, 5.0),
                "stable_env_ratio": 0.6,
            },
        )

        # Selective push robot (30% stable envs)
        self.events.push_robot = EventTerm(
            func=selective_push_by_setting_velocity,
            mode="interval",
            interval_range_s=(6.0, 16.0),
            params={
                "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
                "stable_env_ratio": 0.3,
            },
        )

        # Velocity commands
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.resampling_time_range = (7.0, 13.0)

        # Terminations
        self.terminations.base_contact.params["threshold"] = 0.5

        self.terminations.head_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_upper", "Head_lower"]),
                "threshold": 0.5,
            },
        )

        self.terminations.base_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 2.0},
        )

        # Rewards
        self.rewards.track_lin_vel_xy_exp = RewTerm(
            func=mdp.track_lin_vel_xy_exp, weight=1.5, params={"command_name": "base_velocity", "std": 0.5}
        )
        self.rewards.track_ang_vel_z_exp = RewTerm(
            func=mdp.track_ang_vel_z_exp, weight=0.75, params={"command_name": "base_velocity", "std": 0.5}
        )

        self.rewards.foot_clearance = RewTerm(
            func=foot_clearance_reward,
            weight=0.25,
            params={
                "std": 0.01,
                "tanh_mult": 2.0,
                "target_height": 0.1,
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )

        self.rewards.action_smoothness = RewTerm(
            func=action_smoothness_penalty,
            weight=-0.05,
        )

        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*thigh", ".*calf"]),
                "threshold": 0.5,
            },
        )

        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated,
            weight=-10.0,
        )


@configclass
class UnitreeGo2SequorEnvCfg_PLAY(UnitreeGo2SequorEnvCfg):
    """Play mode configuration for flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.commands.base_velocity.resampling_time_range = (4.0, 4.0)

        self.events.push_robot = None
        self.events.base_external_force_torque = None
        self.events.randomize_actuator_gains = None
        self.curriculum = None

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


# =============================================================================
# Rough terrain environment
# =============================================================================


@configclass
class RoughObservationsCfg:
    """Asymmetric Actor-Critic observations for rough terrain locomotion."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations - excludes base_lin_vel for better sim-to-real transfer."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations - includes base_lin_vel for better value estimation."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RoughCurriculumCfg:
    """Curriculum configuration for rough terrain progressive training."""

    physics_material = CurrTerm(
        func=modify_physics_material_curriculum,
        params={
            "term_name": "physics_material",
            "num_steps": 10000,
            "warmup_steps": 5000,
            "initial_friction_range": (0.8, 0.8),
            "final_friction_range": (0.7, 1.0),
            "initial_restitution_range": (0.0, 0.0),
            "final_restitution_range": (0.0, 0.0),
        },
    )

    external_forces = CurrTerm(
        func=modify_external_forces_curriculum,
        params={
            "term_name": "base_external_force_torque",
            "num_steps": 10000,
            "warmup_steps": 5000,
            "initial_force_range": (0.0, 0.0),
            "final_force_range": (-1.0, 1.0),
            "initial_torque_range": (0.0, 0.0),
            "final_torque_range": (-1.0, 1.0),
        },
    )

    push_robot = CurrTerm(
        func=modify_push_robot_curriculum,
        params={
            "term_name": "push_robot",
            "num_steps": 13000,
            "warmup_steps": 10000,
            "initial_velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
            "final_velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
            "initial_interval": (10.0, 20.0),
            "final_interval": (5.0, 20.0),
        },
    )

    velocity_command = CurrTerm(
        func=modify_velocity_command_curriculum,
        params={
            "command_name": "base_velocity",
            "num_steps": 10000,
            "warmup_steps": 3000,
            "initial_lin_vel_x": (-0.5, 0.5),
            "final_lin_vel_x": (-1.0, 1.0),
            "initial_lin_vel_y": (-0.5, 0.5),
            "final_lin_vel_y": (-1.0, 1.0),
            "initial_ang_vel_z": (-0.5, 0.5),
            "final_ang_vel_z": (-0.5, 0.5),
        },
    )

    mass_randomization = CurrTerm(
        func=modify_mass_randomization_curriculum,
        params={
            "term_name": "add_base_mass",
            "num_steps": 5000,
            "warmup_steps": 1000,
            "initial_mass_range": (-1.0, 1.0),
            "final_mass_range": (-1.0, 3.0),
        },
    )

    feet_air_time_weight = CurrTerm(
        func=modify_reward_weight_curriculum,
        params={
            "reward_term_name": "feet_air_time",
            "num_steps": 5000,
            "warmup_steps": 1000,
            "initial_weight": 0.03,
            "final_weight": 0.005,
            "decay_type": "linear",
        },
    )


@configclass
class UnitreeGo2SequorRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    """Enhanced Go2 rough terrain environment with curriculum-based training."""

    observations: RoughObservationsCfg = RoughObservationsCfg()
    curriculum: RoughCurriculumCfg = RoughCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()

        # Scale down terrains for Go2
        self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.05)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        # Disable height scanner
        self.scene.height_scanner = None

        # Physics material event: reset mode for curriculum
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.8, 0.8),
                "dynamic_friction_range": (0.8, 0.8),
                "restitution_range": (0.0, 0.0),
                "num_buckets": 64,
            },
        )

        # PD gain randomization (+/-5%)
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stiffness_distribution_params": (0.95, 1.05),
                "damping_distribution_params": (0.95, 1.05),
                "operation": "scale",
            },
        )

        # Selective external forces (40% stable envs)
        self.events.base_external_force_torque = EventTerm(
            func=selective_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "force_range": (0.0, 0.0),
                "torque_range": (0.0, 0.0),
                "stable_env_ratio": 0.4,
            },
        )

        # Selective push robot (20% stable envs)
        self.events.push_robot = EventTerm(
            func=selective_push_by_setting_velocity,
            mode="interval",
            interval_range_s=(15.0, 16.0),
            params={
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                "stable_env_ratio": 0.2,
            },
        )

        # Velocity commands
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.resampling_time_range = (3.0, 15.0)

        # Terminations
        self.terminations.base_contact.params["threshold"] = 0.5

        self.terminations.head_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_upper", "Head_lower"]),
                "threshold": 0.5,
            },
        )

        self.terminations.base_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 2.5},
        )

        # Rewards
        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated,
            weight=-10.0,
        )

        self.rewards.feet_air_time.weight = 0.01
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.75
        self.rewards.dof_torques_l2.weight = -0.0002
        self.rewards.dof_acc_l2.weight = -2.5e-7

        # Disable undesired contacts for rough terrain
        self.rewards.undesired_contacts = None


@configclass
class UnitreeGo2SequorRoughEnvCfg_PLAY(UnitreeGo2SequorRoughEnvCfg):
    """Play mode configuration for rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False

        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 5.0),
            params={"velocity_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}},
        )

        self.events.randomize_actuator_gains = None
        self.events.base_external_force_torque = None
        self.curriculum = None

        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        self.scene.height_scanner = None
