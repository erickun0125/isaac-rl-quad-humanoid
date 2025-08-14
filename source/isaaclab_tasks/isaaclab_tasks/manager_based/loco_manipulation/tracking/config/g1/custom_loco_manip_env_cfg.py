# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from .loco_manip_env_cfg import G1LocoManipEnvCfg, ALL_CONTROLLED_JOINTS


@configclass
class CustomLocoManipObservationsCfg:
    """Asymmetric Actor-Critic observations for improved locomanipulation learning."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Actor observations - optimized for sim-to-real transfer."""

        # Base state observations
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
        )
        
        # Command observations
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        left_ee_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_pose"})
        right_ee_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "right_ee_pose"})
        
        # Joint state observations with history for better control
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINTS)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=2
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINTS)},
            noise=Unoise(n_min=-0.5, n_max=0.5),
            history_length=2
        )
        
        # Action history for smoother control
        actions = ObsTerm(func=mdp.last_action, history_length=2)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Critic observations - includes privileged information."""

        # Privileged base state information
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        
        # Command observations
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        left_ee_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "left_ee_pose"})
        right_ee_pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "right_ee_pose"})
        
        # Joint states with history
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINTS)},
            history_length=2
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=ALL_CONTROLLED_JOINTS)},
            history_length=2
        )
        
        # Action history
        actions = ObsTerm(func=mdp.last_action, history_length=2)

        def __post_init__(self):
            self.enable_corruption = False  # Critic gets clean observations
            self.concatenate_terms = True

    # Observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()




@configclass
class G1CustomLocoManipEnvCfg(G1LocoManipEnvCfg):
    """Enhanced G1 locomanipulation environment with improved observations."""
    
    # Override observations with asymmetric actor-critic setup
    observations: CustomLocoManipObservationsCfg = CustomLocoManipObservationsCfg()

    def __post_init__(self):
        # Call parent post init first
        super().__post_init__()

        #--------------------------------
        # Enhanced Events for Robustness
        #--------------------------------
        
        # Enhanced external forces with selective application
        self.events.base_external_force_torque = EventTerm(
            func=mdp.apply_external_force_torque,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names="pelvis"),
                "force_range": (-1.0, 1.0),
                "torque_range": (-1.0, 1.0),
            },
        )
        
        # Mass randomization for payload simulation
        if hasattr(self.events, 'add_base_mass') and self.events.add_base_mass is not None:
            self.events.add_base_mass.params["mass_distribution_params"] = (-1.0, 2.0)  # Simulate carrying loads
        
        # Enhanced hand disturbances for manipulation robustness (using wrist_yaw_link)
        if hasattr(self.events, 'left_hand_force') and self.events.left_hand_force is not None:
            self.events.left_hand_force.params["asset_cfg"] = SceneEntityCfg("robot", body_names="left_wrist_yaw_link")
            self.events.left_hand_force.params["force_range"] = (-1.0, 1.0)
            self.events.left_hand_force.params["torque_range"] = (-1.0, 1.0)
        if hasattr(self.events, 'right_hand_force') and self.events.right_hand_force is not None:
            self.events.right_hand_force.params["asset_cfg"] = SceneEntityCfg("robot", body_names="right_wrist_yaw_link")
            self.events.right_hand_force.params["force_range"] = (-1.0, 1.0)
            self.events.right_hand_force.params["torque_range"] = (-1.0, 1.0)

        #--------------------------------
        # Enhanced Commands
        #--------------------------------
        
        # More diverse velocity commands for better locomotion
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.resampling_time_range = (8.0, 12.0)
        
        # Faster end-effector command changes for dynamic manipulation
        self.commands.left_ee_pose.resampling_time_range = (1.0, 3.0)
        self.commands.right_ee_pose.resampling_time_range = (1.0, 3.0)

        #--------------------------------
        # Enhanced Terminations
        #--------------------------------
        
        # 

        #--------------------------------
        # Enhanced Rewards
        #--------------------------------
        
        # Stronger locomotion tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 2.5
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        
        # Enhanced manipulation tracking with adaptive weights
        self.rewards.left_ee_pos_tracking.weight = -3.0
        self.rewards.left_ee_pos_tracking_fine_grained.weight = 3.0
        self.rewards.right_ee_pos_tracking.weight = -3.0
        self.rewards.right_ee_pos_tracking_fine_grained.weight = 3.0
        
        # Orientation tracking rewards with higher weights
        self.rewards.left_end_effector_orientation_tracking.weight = -0.5
        self.rewards.right_end_effector_orientation_tracking.weight = -0.5
        
        # Enhanced reward weights for better performance
        self.rewards.action_rate_l2.weight = -0.015  # Slightly higher penalty for smoother actions
        self.rewards.joint_accel_l2.weight = -5.0e-7  # Joint acceleration penalty for smoother motion
        self.rewards.joint_torques_l2.weight = -5.0e-5  # Energy efficiency reward
        
        # Stability rewards
        self.rewards.base_height_l2.weight = -2.0
        self.rewards.flat_orientation_l2.weight = -8.0
        
        # Enhanced feet air time for better bipedal walking
        self.rewards.feet_air_time.weight = 0.3
        self.rewards.feet_air_time.params["threshold"] = 0.6
        
        # Termination penalty
        self.rewards.termination_penalty.weight = -150.0


@configclass
class G1CustomLocoManipEnvCfg_PLAY(G1CustomLocoManipEnvCfg):
    """Play mode configuration for testing trained policies."""
    
    def __post_init__(self):
        # Call parent post init
        super().__post_init__()

        # Smaller scene for play
        self.scene.num_envs = 16
        self.scene.env_spacing = 3.0
        
        # Disable noise for clean observations
        self.observations.policy.enable_corruption = False
        
        # Disable disturbances for demonstration
        if hasattr(self.events, 'base_external_force_torque'):
            self.events.base_external_force_torque = None
        if hasattr(self.events, 'left_hand_force'):
            self.events.left_hand_force = None
        if hasattr(self.events, 'right_hand_force'):
            self.events.right_hand_force = None
        
        # Smoother commands for demonstration
        self.commands.base_velocity.resampling_time_range = (5.0, 5.0)
        self.commands.left_ee_pose.resampling_time_range = (2.0, 2.0)
        self.commands.right_ee_pose.resampling_time_range = (2.0, 2.0)
        
        # No curriculum in play mode
        
        # Set moderate command ranges for demonstration
        self.commands.base_velocity.ranges.lin_vel_x = (-0.8, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.6, 0.6)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)