# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recovery-specific reward terms for the MDP."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def target_configuration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    flat_orientation_weight: float = -1.0,  # Weight for flat orientation reward
    joint_pose_weight: float = 1.0,        # Weight for joint pose reward
    final_config_weight: float = 1.0,      # Weight for final configuration reward
    foot_contact_weight: float = 1.0,      # Weight for foot contact reward
    joint_orientation_threshold: float = 0.524,  # radians (30 degrees for joint_pose)
    foot_orientation_threshold_lower: float = 0.471,    # radians (27 degrees, lower bound for foot contact)
    foot_orientation_threshold_upper: float = 0.942,  # radians (54 degrees, upper bound for foot contact)
    final_orientation_threshold: float = 0.087,  # radians (5 degrees for final_configuration)
    final_joint_threshold: float = 0.1,        # radians (6 degrees total for final_configuration)
    final_time_threshold: float = 0.9,          # minimum episode progress (t/T) for final reward
    std: float = 0.5,                      # radians (for joint_pose)
    big_reward: float = 100.0,             # for final_configuration
    contact_force_threshold: float = 0.1,  # threshold for foot contact detection (N)
) -> torch.Tensor:
    """Unified reward function that combines all reward components.
    
    This function computes common calculations once and combines rewards:
    1. Flat orientation reward: Encourages upright orientation (dot_product^2)
    2. Joint pose reward: Conditional joint pose reward when robot is upright enough
    3. Final configuration reward: Binary reward when both orientation and joint conditions are met (with time constraint)
    4. Foot contact reward: Reward based on number of feet in contact when upright
    
    Args:
        env: The environment instance
        asset_cfg: Scene entity configuration for the robot
        flat_orientation_weight: Weight for flat orientation component
        joint_pose_weight: Weight for joint pose component
        final_config_weight: Weight for final configuration component
        foot_contact_weight: Weight for foot contact component
        joint_orientation_threshold: Threshold for joint_pose orientation check (radians)
        foot_orientation_threshold_lower: Lower threshold for foot_contact orientation check (radians)
        foot_orientation_threshold_upper: Upper threshold for foot_contact orientation check (radians)
        final_orientation_threshold: Threshold for final_config orientation check (radians)
        final_joint_threshold: Maximum sum of joint position errors for final_config (radians)
        final_time_threshold: Minimum episode progress (t/T) required for final_config reward
        std: Standard deviation for joint position error normalization (radians)
        big_reward: Large reward value for final_configuration
        contact_force_threshold: Threshold for foot contact detection (N)
        
    Returns:
        Combined reward tensor for each environment
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # === COMMON CALCULATIONS (computed once) ===
    
    # Get robot root orientation quaternion (world frame)
    root_quat_w = robot.data.root_quat_w  # [num_envs, 4] (w, x, y, z)
    
    # Extract z-axis from quaternion (robot's up direction in world frame)
    robot_z_axis = math_utils.quat_apply(root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1))
    
    # World z-axis (gravity opposite direction)
    world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)
    
    # Calculate dot product (1 when upright, 0 when flat)
    dot_product = torch.sum(robot_z_axis * world_z_axis, dim=1)  # [num_envs]
    dot_product = torch.clamp(dot_product, -1.0, 1.0)  # Numerical stability
    
    # Calculate angle between robot z-axis and world z-axis
    angle_rad = torch.acos(dot_product)  # [num_envs]
    
    # Get current and target joint positions
    current_joint_pos = robot.data.joint_pos  # [num_envs, num_joints]
    
    # Hardcoded target joint positions for recovery task
    target_values = [0.35, -0.35, 0.5, -0.5, 1.36, 1.36, 1.36, 1.36, -2.65, -2.65, -2.65, -2.65]
    target_joint_pos = torch.tensor(target_values, device=env.device).expand(env.num_envs, -1)  # [num_envs, num_joints]
    
    # Calculate absolute joint position error
    joint_pos_error_abs = torch.abs(current_joint_pos - target_joint_pos)  # [num_envs, num_joints]
    joint_pos_error_abs_max = torch.max(joint_pos_error_abs, dim=1)[0]  # [num_envs]
    joint_pos_error_square = torch.square(joint_pos_error_abs)  # [num_envs, num_joints]
    #joint_pos_error_square_mean = torch.mean(joint_pos_error_square, dim=1)  # [num_envs]

    # === CALCULATE ALL THREE REWARD COMPONENTS ===
    
    # 1. Flat orientation reward: squared dot product (1 when upright, 0 when flat)
    flat_orientation_reward_1 = 1. - dot_product #2~0
    flat_orientation_reward_2 = angle_rad #3.14~0
    flat_orientation_reward = flat_orientation_reward_1 + 0.1 *flat_orientation_reward_2 **2

    # 2. Joint pose reward: conditional on orientation threshold
    is_upright_for_joint_pose = angle_rad <= joint_orientation_threshold  # [num_envs]
    
    # Calculate joint position reward component for hip joints only (first 4 joints)
    joint_pos_reward_per_joint = torch.exp(-0.5 * (joint_pos_error_square / std**2))  # [num_envs, num_joints]
    hip_joint_pos_reward = joint_pos_reward_per_joint[:, :4]  # [num_envs, 4] - only hip joints
    non_hip_joint_pos_reward = joint_pos_reward_per_joint[:, 4:]  # [num_envs, 8] - exclude hip joints

    joint_pos_reward_component = torch.mean(hip_joint_pos_reward, dim=1) + 2.0 *torch.mean(non_hip_joint_pos_reward, dim=1) # [num_envs] - mean across hip joints

    joint_pose_reward = torch.where(is_upright_for_joint_pose, joint_pos_reward_component,
                                   0.0 * joint_pos_reward_component)

    # 3. Foot contact reward: reward based on number of feet in contact when upright
    is_upright_for_foot_contact = (angle_rad >= foot_orientation_threshold_lower) & (angle_rad <= foot_orientation_threshold_upper)  # [num_envs]
    contact_sensor = env.scene["contact_forces"]
    contact_forces = contact_sensor.data.net_forces_w_history  # [num_envs, num_bodies, history_length, 3]
    
    # Get foot contact forces (assume last 4 bodies are feet: FL, FR, RL, RR)
    foot_forces = contact_forces[:, -4:, -1, :]  # [num_envs, 4, 3] - latest contact forces for 4 feet
    foot_contact_magnitude = torch.norm(foot_forces, dim=2)  # [num_envs, 4] - magnitude of contact force per foot
    
    # Determine which feet are in contact
    feet_in_contact = foot_contact_magnitude > contact_force_threshold  # [num_envs, 4] - boolean mask
    num_feet_in_contact = torch.sum(feet_in_contact.float(), dim=1)  # [num_envs] - count of feet in contact
    
    # Give reward based on number of feet in contact, but only when upright
    foot_contact_reward_component = num_feet_in_contact / 4.0  # [num_envs] - normalized to 0-1 range
    foot_contact_reward = torch.where(is_upright_for_foot_contact, foot_contact_reward_component,
                                     torch.zeros_like(foot_contact_reward_component))

    # 4. Final configuration reward: exponential reward based on non-hip joints when conditions are met
    current_step = env.episode_length_buf.float()  # [num_envs] - current episode step
    max_steps = float(env.max_episode_length)      # maximum episode length
    progress_ratio = current_step / max_steps      # [num_envs] - normalized progress (0-1)
    
    orientation_satisfied = angle_rad <= final_orientation_threshold  # [num_envs]
    
    # Check only hip joint errors for joint satisfaction condition
    hip_joint_pos_error_abs = joint_pos_error_abs[:, :4]  # [num_envs, 4] - only hip joints
    hip_joint_pos_error_abs_max = torch.max(hip_joint_pos_error_abs, dim=1)[0]  # [num_envs]
    joint_satisfied = hip_joint_pos_error_abs_max <= final_joint_threshold  # [num_envs]
    
    time_satisfied = progress_ratio >= final_time_threshold  # [num_envs] - only reward after sufficient time
    all_satisfied = orientation_satisfied & joint_satisfied & time_satisfied  # [num_envs]
    
    # Calculate exponential reward based on non-hip joints (thigh and calf joints)
    final_joint_pos_reward_per_joint = torch.exp(-0.5 * (joint_pos_error_square / (std / 4.0)**2))  # [num_envs, num_joints]
    final_joint_pos_reward_component = torch.mean(final_joint_pos_reward_per_joint, dim=1)  # [num_envs] - mean across non-hip joints
    
    final_config_reward = torch.where(all_satisfied, 
                                     final_joint_pos_reward_component,  # Scale exponential reward by big_reward
                                     torch.zeros(env.num_envs, device=env.device))
    
    # === COMBINE ALL REWARDS ===
    total_reward = (flat_orientation_weight * flat_orientation_reward + 
                   joint_pose_weight * joint_pose_reward + 
                   final_config_weight * final_config_reward +
                   foot_contact_weight * foot_contact_reward)
    
    return total_reward


# Convenience wrapper functions for backward compatibility
def flat_orientation(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for upright orientation (squared dot product)."""
    return target_configuration(env, asset_cfg, 
                               flat_orientation_weight=-1.0, 
                               joint_pose_weight=0.0, 
                               final_config_weight=0.0,
                               foot_contact_weight=0.0)


def joint_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for reaching target joint pose when robot orientation is upright."""
    return target_configuration(env, asset_cfg, 
                               flat_orientation_weight=0.0, 
                               joint_pose_weight=1.0, 
                               final_config_weight=0.0,
                               foot_contact_weight=0.0,
                               joint_orientation_threshold=0.524,  # 30 degrees
                               std=0.2,  # radians (for joint_pose
                               )

def foot_contact(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for reaching target joint pose when robot orientation is upright."""
    return target_configuration(env, asset_cfg, 
                               flat_orientation_weight=0.0, 
                               joint_pose_weight=0.0, 
                               final_config_weight=0.0,
                               foot_contact_weight=1.0,
                               foot_orientation_threshold_lower=0.471,   # 27 degrees
                               foot_orientation_threshold_upper=0.942,  # 54 degrees
                               )

def final_configuration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Final configuration reward for successful recovery."""
    return target_configuration(env, asset_cfg, 
                               flat_orientation_weight=0.0, 
                               joint_pose_weight=0.0, 
                               final_config_weight=1.0,
                               foot_contact_weight=0.0,
                               final_orientation_threshold=0.16,  # 10 degrees
                               final_joint_threshold=0.628,  # 36 degrees total
                               final_time_threshold=0.8,   # only after 30% of episode
                               std=0.2,  # radians (for joint_pose)
                               big_reward=100.0,  # Large bonus for achieving final configuration
                               )


def ang_vel_xy_threshold_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 2.0,  # rad/s threshold for angular velocity
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for angular velocity in xy-plane above threshold.
    
    Args:
        env: The environment instance
        threshold: Angular velocity threshold (rad/s). No penalty below this value.
        asset_cfg: Scene entity configuration for the robot
        
    Returns:
        Penalty tensor (0 when below threshold, -squared_excess when above)
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get xy angular velocity magnitude
    ang_vel_xy = robot.data.root_ang_vel_b[:, :2]  # [num_envs, 2] (x, y components)
    ang_vel_xy_magnitude = torch.norm(ang_vel_xy, dim=1)  # [num_envs]
    
    # Calculate excess above threshold
    excess = ang_vel_xy_magnitude - threshold  # [num_envs]
    excess = torch.clamp(excess, min=0.0)  # Only positive excess
    
    # Square the excess for smooth penalty
    penalty = torch.square(excess)  # Negative because it's a penalty
    
    return penalty


def joint_vel_threshold_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 5.0,  # rad/s threshold for joint velocity
    early_episode_multiplier: float = 10.0,  # penalty multiplier for early episode
    progress_threshold: float = 0.05,  # episode progress below which to apply extra penalty
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for joint velocities above threshold, with higher penalty in early episode.
    
    Args:
        env: The environment instance
        threshold: Joint velocity threshold (rad/s). No penalty below this value.
        early_episode_multiplier: Multiplier for penalty when progress_ratio < progress_threshold
        progress_threshold: Episode progress below which to apply extra penalty (0-1)
        asset_cfg: Scene entity configuration for the robot
        
    Returns:
        Penalty tensor (0 when below threshold, -squared_excess when above, higher penalty early)
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # Get episode progress
    current_step = env.episode_length_buf.float()  # [num_envs] - current episode step
    max_steps = float(env.max_episode_length)      # maximum episode length
    progress_ratio = current_step / max_steps      # [num_envs] - normalized progress (0-1)
    
    # Get absolute joint velocities
    joint_vel_abs = torch.abs(robot.data.joint_vel)  # [num_envs, num_joints]
    
    # Calculate excess above threshold for each joint
    excess = joint_vel_abs - threshold  # [num_envs, num_joints]
    excess = torch.clamp(excess, min=0.0)  # Only positive excess
    
    # Square the excess and sum across joints
    excess_squared = torch.square(excess)  # [num_envs, num_joints]
    base_penalty = torch.sum(excess_squared, dim=1)  # [num_envs] - sum across joints
    '''
    # Apply higher penalty in early episode
    is_early_episode = progress_ratio < progress_threshold  # [num_envs] - boolean mask
    penalty_multiplier = torch.where(is_early_episode, 
                                    torch.full_like(progress_ratio, early_episode_multiplier),
                                    torch.ones_like(progress_ratio))  # [num_envs]
    
    total_penalty = base_penalty * penalty_multiplier  # [num_envs] - apply multiplier
    '''
    total_penalty = base_penalty  # [num_envs] - no multiplier applied
    return total_penalty


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]

    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward