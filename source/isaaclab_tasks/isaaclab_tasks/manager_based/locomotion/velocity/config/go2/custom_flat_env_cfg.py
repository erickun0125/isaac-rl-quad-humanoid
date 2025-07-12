# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
from .curriculums import *  # Import curriculum functions
from .rewards import nominal_joint_pos_when_static_reward  # Import Go2 specific reward functions
from .events import selective_external_force_torque, selective_push_by_setting_velocity  # Import Go2 specific event functions
from .flat_env_cfg import UnitreeGo2FlatEnvCfg


@configclass
class CustomObservationsCfg:
    """Asymmetric Actor-Critic observations for improved locomotion learning."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations - excludes base_lin_vel for better sim-to-real transfer."""

        # observation terms (order preserved)
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

        # observation terms (order preserved) 
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)  # Critic gets privileged info
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False  # Critic gets clean observations
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()      # Actor용 (45차원)
    critic: CriticCfg = CriticCfg()      # Critic용 (48차원)


@configclass  
class CustomCurriculumCfg:
    """Curriculum configuration for progressive training difficulty."""
    
    # Physics material curriculum
    physics_material = CurrTerm(
        func=modify_physics_material_curriculum,
        params={
            "term_name": "physics_material",
            "num_steps": 10000,  # 통일된 duration
            "warmup_steps": 1000,  # Warmup 기간
            "initial_friction_range": (0.8, 0.8),  # 실제 Go2 기본값
            "final_friction_range": (0.5, 1.3),
            "initial_restitution_range": (0.0, 0.0),  # 실제 Go2 기본값
            "final_restitution_range": (0.0, 0.0),
        }
    )
    
    # External forces curriculum
    external_forces = CurrTerm(
        func=modify_external_forces_curriculum,
        params={
            "term_name": "base_external_force_torque",
            "num_steps": 10000,  # 통일된 duration
            "warmup_steps": 2000,  # Warmup 기간
            "initial_force_range": (0.0, 0.0),  # 실제 Go2 기본값
            "final_force_range": (-10.0, 10.0),
            "initial_torque_range": (0.0, 0.0),  # 실제 Go2 기본값
            "final_torque_range": (-10.0, 10.0),
        }
    )
    
    # Push robot curriculum
    push_robot = CurrTerm(
        func=modify_push_robot_curriculum,
        params={
            "term_name": "push_robot",
            "num_steps": 10000,  # 통일된 duration
            "warmup_steps": 1000,  # Warmup 기간
            "initial_velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},  # Go2에서 push_robot=None이므로 0부터 시작
            "final_velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            "initial_interval": (11.0, 20.0),
            "final_interval": (5.0, 10.0),
        }
    )
    
    # Velocity command curriculum  
    velocity_command = CurrTerm(
        func=modify_velocity_command_curriculum,
        params={
            "command_name": "base_velocity",
            "num_steps": 5000,  # 통일된 duration
            "warmup_steps": 1000,  # Warmup 기간
            "initial_lin_vel_x": (-1.0, 1.0),  # 실제 Go2 기본값
            "final_lin_vel_x": (-3.0, 3.0),
            "initial_lin_vel_y": (-1.0, 1.0),  # 실제 Go2 기본값
            "final_lin_vel_y": (-2.0, 2.0),
            "initial_ang_vel_z": (-1.0, 1.0),  # 실제 Go2 기본값
            "final_ang_vel_z": (-2.0, 2.0),
        }
    )
    
    # Mass randomization curriculum
    mass_randomization = CurrTerm(
        func=modify_mass_randomization_curriculum,
        params={
            "term_name": "add_base_mass",
            "num_steps": 5000,  # 통일된 duration
            "warmup_steps": 1000,  # Warmup 기간
            "initial_mass_range": (-1.0, 3.0),  # 실제 Go2 기본값 (수정됨)
            "final_mass_range": (-1.0, 8.0),
        }
    )
    
    # Feet air time reward weight curriculum
    feet_air_time_weight = CurrTerm(
        func=modify_reward_weight_curriculum,
        params={
            "reward_term_name": "feet_air_time",
            "num_steps": 5000,  # Curriculum 지속 기간
            "warmup_steps": 1000,  # 초기 2000 iterations는 full weight 유지
            "initial_weight": 0.125,  # 원래 설정값 (velocity_env_cfg.py 기본값)
            "final_weight": 0.05,      # 최종적으로 완전히 제거
            "decay_type": "cosine",   # 부드러운 cosine 감소
        }
    )


@configclass
class UnitreeGo2SequorEnvCfg(UnitreeGo2FlatEnvCfg):
    """Enhanced Go2 environment with curriculum-based training."""
    
    # Override observations with asymmetric actor-critic setup
    observations: CustomObservationsCfg = CustomObservationsCfg()
    
    # Add custom curriculum configuration
    curriculum: CustomCurriculumCfg = CustomCurriculumCfg()

    def __post_init__(self):
        # Call parent post init first (inherits all existing events)
        super().__post_init__()

        #--------------------------------
        
        # ✅ 기존 events를 상속받은 후 필요한 것만 수정/추가
        
        # Physics material event를 "reset" 모드로 재정의 (curriculum 작동을 위해)
        self.events.physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="reset",  # "startup"에서 "reset"으로 변경
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.8, 0.8),  # 초기값 (curriculum에서 수정됨)
                "dynamic_friction_range": (0.8, 0.8),  # 초기값 (curriculum에서 수정됨)
                "restitution_range": (0.0, 0.0),      # 초기값 (curriculum에서 수정됨)
                "num_buckets": 64,
            },
        )
        
        # PD gain randomization 추가 (±5% randomization)
        self.events.randomize_actuator_gains = EventTerm(
            func=mdp.randomize_actuator_gains,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stiffness_distribution_params": (0.95, 1.05),  # ±5%
                "damping_distribution_params": (0.95, 1.05),     # ±5%
                "operation": "scale",
            },
        )
        
        # External forces를 selective로 변경 (50%의 로봇은 영향 안 받음)
        self.events.base_external_force_torque = EventTerm(
            func=selective_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="base"),
                "force_range": (0.0, 0.0),    # 초기값 (curriculum에서 수정됨)
                "torque_range": (0.0, 0.0),   # 초기값 (curriculum에서 수정됨)
                "stable_env_ratio": 0.6,      
            },
        )
        
        # Push robot을 selective로 활성화 (Go2에서는 기본적으로 None이므로 활성화 필요)
        self.events.push_robot = EventTerm(
            func=selective_push_by_setting_velocity,
            mode="interval",
            interval_range_s=(15.0, 16.0),  # 초기 간격 (curriculum에서 조정됨)
            params={
                "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},  # 초기값 (curriculum에서 조정됨)
                "stable_env_ratio": 0.3,  
            },
        )

        #--------------------------------
        
        # Velocity command에서 10%의 로봇이 항상 정지하도록 설정
        self.commands.base_velocity.rel_standing_envs = 0.1

        #--------------------------------

        # Base contact threshold를 조절
        self.terminations.base_contact.params["threshold"] = 0.5
        
        # Orientation 기반 termination 추가 (로봇이 많이 기울어지면 넘어진 것으로 간주)
        self.terminations.base_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 2.0},  # (라디안)
        )
        
        #--------------------------------
        
        # Termination penalty 추가 (넘어져서 episode 조기 종료 시 큰 penalty)
        self.rewards.termination_penalty = RewTerm(
            func=mdp.is_terminated,
            weight=-10.0,  # 큰 음수 계수 = 큰 penalty
        )

        # 정지 상태에서 nominal joint position 유지 reward 추가
        self.rewards.nominal_joint_pos_when_static = RewTerm(
            func=nominal_joint_pos_when_static_reward,
            weight=1.0,  # 정지 상태에서만 활성화되므로 적당한 weight
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
                "stand_still_scale": 1.0, 
            }
        ) 


@configclass
class UnitreeGo2SequorEnvCfg_PLAY(UnitreeGo2SequorEnvCfg):
    """Play mode configuration without curriculum and disturbances."""
    
    def __post_init__(self):
        # Call parent post init
        super().__post_init__()

        # Smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # Spawn randomly in grid
        self.scene.terrain.max_init_terrain_level = None
        
        # Reduce terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # Disable noise for play
        self.observations.policy.enable_corruption = False
        
        # interval
        self.events.push_robot = EventTerm(
            func=mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 5.0),
            params={"velocity_range": {"x": (-2.0, 2.0), "y": (-2.0, 2.0)}},
        )

        #self.events.push_robot = None
        self.events.randomize_actuator_gains = None  # No PD gain randomization for play
        
        # Disable curriculum
        self.curriculum = None
        
        # Set final velocity ranges for play
        self.commands.base_velocity.ranges.lin_vel_x = (-0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0) 