# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP curriculum functions for G1 loco-manipulation tasks."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import SceneEntityCfg

#Curriculums for Loco-Manipulation

def velocity_range_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    lin_reward_term: str = "track_lin_vel_xy_exp",
    ang_reward_term: str = "track_ang_vel_z_exp",
    lin_vel_x_limit: list[float] | None = None,
    lin_vel_y_limit: list[float] | None = None,
    ang_vel_z_limit: list[float] | None = None,
    reward_threshold: float = 0.75,
    delta_step: float = 0.1,
) -> torch.Tensor:
    """Combined curriculum for both linear and angular velocity commands.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to apply curriculum to.
        lin_reward_term: Linear velocity tracking reward term name.
        ang_reward_term: Angular velocity tracking reward term name.
        lin_vel_x_limit: [min, max] limits for linear velocity x range.
        lin_vel_y_limit: [min, max] limits for linear velocity y range.
        ang_vel_z_limit: [min, max] limits for angular velocity z range.
        reward_threshold: Combined reward threshold to trigger curriculum.
        delta_step: Step size for expanding velocity ranges.
    
    Returns:
        Combined velocity curriculum progress metric.
    """
    if lin_vel_x_limit is None:
        lin_vel_x_limit = [-1.5, 1.5]
    if lin_vel_y_limit is None:
        lin_vel_y_limit = [-1.0, 1.0]
    if ang_vel_z_limit is None:
        ang_vel_z_limit = [-2.0, 2.0]
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges

    # Get both reward terms
    lin_reward_term_cfg = env.reward_manager.get_term_cfg(lin_reward_term)
    ang_reward_term_cfg = env.reward_manager.get_term_cfg(ang_reward_term)
    
    lin_reward = torch.mean(env.reward_manager._episode_sums[lin_reward_term][env_ids]) / env.max_episode_length_s
    ang_reward = torch.mean(env.reward_manager._episode_sums[ang_reward_term][env_ids]) / env.max_episode_length_s
    
    # Combined reward performance
    combined_reward = (lin_reward / abs(lin_reward_term_cfg.weight) + ang_reward / abs(ang_reward_term_cfg.weight)) / 2.0

    # Update curriculum every episode
    if env.common_step_counter % env.max_episode_length == 0:
        if combined_reward > reward_threshold:
            # Expand velocity ranges
            delta_command = torch.tensor([-delta_step, delta_step], device=env.device)
            
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                lin_vel_x_limit[0],
                lin_vel_x_limit[1],
            ).tolist()
            
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                lin_vel_y_limit[0],
                lin_vel_y_limit[1],
            ).tolist()
            
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                ang_vel_z_limit[0],
                ang_vel_z_limit[1],
            ).tolist()

    return torch.tensor(combined_reward, device=env.device)


def reward_weight_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    reward_term_names: list[str] | None = None,
    tracking_reward_name: str = "track_lin_vel_xy_exp",
    min_ratio: float = 0.1,
    max_ratio: float = 1.0,
    reward_threshold: float = 0.7,
    ratio_step: float = 0.05,
) -> torch.Tensor:
    """Curriculum to adjust reward weights based on tracking performance using a ratio system.
    
    This function adjusts weights using a ratio system:
    - Original weight values from config are multiplied by a ratio (min_ratio ~ max_ratio)
    - If performance is good: ratio changes by ratio_step (clipped to min_ratio ~ max_ratio)
    - If performance is poor: ratio remains unchanged (no rollback)
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to apply curriculum to.
        reward_term_names: List of reward term names to modify.
        tracking_reward_name: Name of the tracking reward term to monitor.
        min_ratio: Minimum ratio for weight scaling.
        max_ratio: Maximum ratio for weight scaling.
        reward_threshold: Reward threshold as ratio of reward weight to trigger curriculum.
        ratio_step: Step size for adjusting the weight ratio (can be positive or negative).
    
    Returns:
        Current weight ratio for the first reward term.
    """
    if reward_term_names is None:
        reward_term_names = ["joint_deviation_hip", "joint_deviation_arms", "joint_deviation_torso"]
    
    tracking_reward_term = env.reward_manager.get_term_cfg(tracking_reward_name)
    tracking_reward = torch.mean(env.reward_manager._episode_sums[tracking_reward_name][env_ids]) / env.max_episode_length_s

    # Initialize curriculum state storage if not exists
    if not hasattr(env, '_reward_weight_curriculum_state'):
        env._reward_weight_curriculum_state = {}

    # Update curriculum every episode
    if env.common_step_counter % env.max_episode_length == 0:
        # Calculate performance ratio
        performance_ratio = tracking_reward / abs(tracking_reward_term.weight)
        
        for term_name in reward_term_names:
            term_idx = env.reward_manager._term_names.index(term_name) if term_name in env.reward_manager._term_names else -1
            if term_idx >= 0:
                term_cfg = env.reward_manager._term_cfgs[term_idx]
                
                # Initialize state for this term if not exists
                if term_name not in env._reward_weight_curriculum_state:
                    # Set initial ratio based on ratio_step direction
                    if ratio_step >= 0:
                        initial_ratio = min_ratio  # Start from minimum if increasing
                    else:
                        initial_ratio = max_ratio  # Start from maximum if decreasing
                    
                    env._reward_weight_curriculum_state[term_name] = {
                        'original_weight': term_cfg.weight,
                        'current_ratio': initial_ratio
                    }
                    
                    # Apply initial ratio to the weight
                    term_cfg.weight = term_cfg.weight * initial_ratio
                
                state = env._reward_weight_curriculum_state[term_name]
                current_ratio = state['current_ratio']
                
                if performance_ratio > reward_threshold:
                    # Good performance: adjust ratio by ratio_step (can be positive or negative)
                    new_ratio = current_ratio + ratio_step
                    # Clip to valid range
                    new_ratio = max(min(new_ratio, max_ratio), min_ratio)
                    
                    # Update state and apply new weight
                    state['current_ratio'] = new_ratio
                    term_cfg.weight = state['original_weight'] * new_ratio
                # Poor performance: maintain current ratio (no change)

    # Return current ratio of first reward term
    if reward_term_names[0] in env._reward_weight_curriculum_state:
        return torch.tensor(env._reward_weight_curriculum_state[reward_term_names[0]]['current_ratio'], device=env.device)
    else:
        return torch.tensor(1.0, device=env.device)


def disturbance_range_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    termination_term_names: list[str] | None = None,
    reward_term_name: str = "track_lin_vel_xy_exp",
    push_event_names: list[str] | None = None,
    external_force_event_names: list[str] | None = None,
    max_push_velocity: float = 0.5,
    max_force_magnitude: float = 5.0,
    max_torque_magnitude: float = 2.0,
    termination_threshold: float = 0.1,
    reward_threshold: float = 0.8,
    push_increase_step: float = 0.05,
    force_increase_step: float = 0.2,
) -> torch.Tensor:
    """Unified curriculum to increase both push force and external force/torque intensity based on termination rate and reward performance.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to apply curriculum to.
        termination_term_names: List of termination terms to monitor (excluding timeout).
        reward_term_name: Reward term to monitor for performance threshold.
        push_event_names: List of push velocity event names to modify.
        external_force_event_names: List of external force/torque event names to modify.
        max_push_velocity: Maximum push velocity magnitude.
        max_force_magnitude: Maximum external force magnitude.
        max_torque_magnitude: Maximum external torque magnitude.
        termination_threshold: Maximum termination rate to trigger curriculum progression.
        reward_threshold: Minimum reward performance (as ratio of reward weight) to trigger curriculum.
        push_increase_step: Step size for increasing push velocity.
        force_increase_step: Step size for increasing external force/torque magnitude.
    
    Returns:
        Combined disturbance intensity metric.
    """
    if termination_term_names is None:
        termination_term_names = ["base_height", "base_orientation"]
    if push_event_names is None:
        push_event_names = ["push_base"]
    if external_force_event_names is None:
        external_force_event_names = ["torso_wrench"]
        
    # Calculate termination rate (excluding timeout)
    total_terminations = 0
    total_episodes = len(env_ids)
    
    for term_name in termination_term_names:
        if hasattr(env.termination_manager, '_episode_sums') and term_name in env.termination_manager._episode_sums:
            # Count episodes that terminated due to this specific condition
            terminations = torch.sum(env.termination_manager._episode_sums[term_name][env_ids] > 0)
            total_terminations += terminations.item()
    
    # Calculate termination rate
    if total_episodes > 0:
        termination_rate = total_terminations / total_episodes
    else:
        termination_rate = 0.0

    # Calculate reward performance
    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward_performance = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    current_push_magnitude = 0.1  # Default push velocity
    current_force_magnitude = 1.0  # Default external force
    
    # Update curriculum every episode - increase disturbance if both conditions are met:
    # 1. Termination rate is low (robot is stable)
    # 2. Reward performance is high (robot is performing well)
    update_curriculum = (env.common_step_counter % env.max_episode_length == 0 and 
                        termination_rate < termination_threshold and
                        reward_performance > reward_term.weight * reward_threshold)
    
    # Update push velocity events
    for event_name in push_event_names:
        try:
            event_cfg = env.event_manager.get_term_cfg(event_name)
            
            if update_curriculum:
                # Increase push velocity range
                current_range = event_cfg.params.get("velocity_range", {
                    "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (-0.1, 0.1),
                    "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1)
                })
                
                for key in current_range:
                    min_val, max_val = current_range[key]
                    new_min = max(-max_push_velocity, min_val - push_increase_step)
                    new_max = min(max_push_velocity, max_val + push_increase_step)
                    current_range[key] = (new_min, new_max)
                
                event_cfg.params["velocity_range"] = current_range
            
            # Get current push velocity magnitude
            velocity_range = event_cfg.params.get("velocity_range", {"x": (-0.1, 0.1)})
            current_push_magnitude = velocity_range["x"][1]
        except ValueError:
            # Event not found, skip
            continue
    
    # Update external force/torque events
    for event_name in external_force_event_names:
        try:
            event_cfg = env.event_manager.get_term_cfg(event_name)
            
            if update_curriculum:
                # Increase force and torque ranges
                current_force_range = event_cfg.params.get("force_range", (-1.0, 1.0))
                current_torque_range = event_cfg.params.get("torque_range", (-1.0, 1.0))
                
                # Update force range
                force_min, force_max = current_force_range
                new_force_min = max(-max_force_magnitude, force_min - force_increase_step)
                new_force_max = min(max_force_magnitude, force_max + force_increase_step)
                event_cfg.params["force_range"] = (new_force_min, new_force_max)
                
                # Update torque range
                torque_min, torque_max = current_torque_range
                new_torque_min = max(-max_torque_magnitude, torque_min - force_increase_step * 0.5)
                new_torque_max = min(max_torque_magnitude, torque_max + force_increase_step * 0.5)
                event_cfg.params["torque_range"] = (new_torque_min, new_torque_max)
            
            # Get current force magnitude
            current_force_range = event_cfg.params.get("force_range", (-1.0, 1.0))
            current_force_magnitude = current_force_range[1]
        except ValueError:
            # Event not found, skip
            continue
    
    # Return combined disturbance metric (normalized sum of push and force magnitudes)
    combined_disturbance = (current_push_magnitude / max_push_velocity + current_force_magnitude / max_force_magnitude) / 2.0
    return torch.tensor(combined_disturbance, device=env.device)



#Rewards for Loco-Manipulation

def foot_clearance_reward(
    env: "ManagerBasedRLEnv",
    asset_cfg: "SceneEntityCfg",
    target_height: float = 0.05,
    std: float = 0.1,
    tanh_mult: float = 2.0,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground.
    
    This reward encourages proper foot lifting during swing phase by:
    1. Measuring the error between current foot height and target clearance height
    2. Weighting this error by foot horizontal velocity (higher velocity = swing phase)
    3. Returning exponential reward that peaks when feet are at target height during swing
    
    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the robot feet.
        target_height: Target clearance height for swinging feet (meters).
        std: Standard deviation for the exponential reward function.
        tanh_mult: Multiplier for tanh function applied to foot velocity.
    
    Returns:
        Foot clearance reward for each environment.
    """
    # Get the robot asset
    asset = env.scene[asset_cfg.name]
    
    # Get foot positions in world frame (z-coordinate is height)
    foot_positions_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]  # [num_envs, num_feet, 3]
    foot_heights = foot_positions_w[:, :, 2]  # [num_envs, num_feet]
    
    # Get foot velocities in world frame
    foot_velocities_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]  # [num_envs, num_feet, 3]
    foot_horizontal_vel = foot_velocities_w[:, :, :2]  # [num_envs, num_feet, 2] (x, y components)
    
    # Calculate horizontal velocity magnitude for each foot
    foot_vel_magnitude = torch.norm(foot_horizontal_vel, dim=2)  # [num_envs, num_feet]
    
    # Apply tanh to velocity to get swing phase indicator (0 = stance, 1 = swing)
    swing_phase_indicator = torch.tanh(tanh_mult * foot_vel_magnitude)  # [num_envs, num_feet]
    
    # Calculate height error from target clearance height
    height_error = torch.square(foot_heights - target_height)  # [num_envs, num_feet]
    
    # Weight height error by swing phase (only penalize during swing)
    weighted_error = height_error * swing_phase_indicator  # [num_envs, num_feet]
    
    # Sum error across all feet and apply exponential reward
    total_error = torch.sum(weighted_error, dim=1)  # [num_envs]
    reward = torch.exp(-total_error / std)  # [num_envs]
    
    return reward


def torso_orientation_l2(
    env: "ManagerBasedRLEnv",
    asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize non-flat torso orientation using L2 squared kernel.
    
    This is computed by penalizing the xy-components of the projected gravity vector
    for the specified torso body (e.g., torso_link).
    
    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the torso body.
    
    Returns:
        Orientation penalty for each environment.
    """
    # Get the robot asset
    asset = env.scene[asset_cfg.name]
    
    # Get torso body orientation in world frame
    torso_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]  # [num_envs, 4]
    
    # Convert quaternion to rotation matrix to get gravity projection
    
    # Transform gravity to torso body frame
    # For quaternion [w, x, y, z], we need to rotate the gravity vector
    quat_w, quat_x, quat_y, quat_z = torso_quat_w[:, 0], torso_quat_w[:, 1], torso_quat_w[:, 2], torso_quat_w[:, 3]
    
    # Rotation matrix from quaternion (world to body frame)
    # We only need the third column of rotation matrix (z-axis in body frame)
    r13 = 2 * (quat_x * quat_z + quat_w * quat_y)  # x-component of z-axis in body frame
    r23 = 2 * (quat_y * quat_z - quat_w * quat_x)  # y-component of z-axis in body frame
    r33 = quat_w**2 - quat_x**2 - quat_y**2 + quat_z**2  # z-component of z-axis in body frame
    
    # Projected gravity in body frame (gravity dot with body z-axis)
    projected_gravity_b = torch.stack([r13, r23, r33], dim=1)
    
    # Penalize xy-components (should be close to zero for flat orientation)
    return torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)

def torso_backward_tilt_penalty(
    env: "ManagerBasedRLEnv",
    asset_cfg: "SceneEntityCfg",
) -> torch.Tensor:
    """Penalize backward tilt (negative pitch) of the torso using L2 squared kernel.
    
    This function only penalizes when the torso tilts backward (negative rotation around y-axis
    in body frame). Forward tilt (positive pitch) is allowed and not penalized.
    
    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the torso body.
    
    Returns:
        Backward tilt penalty for each environment.
    """
    # Get the robot asset
    asset = env.scene[asset_cfg.name]
    
    # Get torso body orientation in world frame
    torso_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0], :]  # [num_envs, 4]
    
    # Extract quaternion components [w, x, y, z]
    quat_w, quat_x, quat_y, quat_z = torso_quat_w[:, 0], torso_quat_w[:, 1], torso_quat_w[:, 2], torso_quat_w[:, 3]
    
    # Calculate pitch angle from quaternion
    # pitch = atan2(2*(w*y - x*z), 1 - 2*(y^2 + z^2))
    # For small angles, we can use the approximation: pitch ≈ 2*(w*y - x*z)
    pitch_approx = 2 * (quat_w * quat_y - quat_x * quat_z)
    
    # Alternative: More accurate pitch calculation
    # sin_pitch = 2 * (quat_w * quat_y - quat_x * quat_z)
    # cos_pitch = 1 - 2 * (quat_y**2 + quat_z**2)
    # pitch = torch.atan2(sin_pitch, cos_pitch)
    
    # Only penalize negative pitch (backward tilt)
    # Use ReLU to only consider negative values (backward tilt)
    backward_tilt = torch.clamp(-pitch_approx, min=0.0)  # Only negative pitch values
    
    # Return L2 squared penalty
    return backward_tilt

# Observations for Loco-Manipulation

def foot_contact(
    env: "ManagerBasedRLEnv",
    sensor_name: str = "contact_forces",
    threshold: float = 1.0,
) -> torch.Tensor:
    """Check if the feet are in contact with the ground.
    
    This function returns binary contact information for left and right feet
    based on contact force measurements from the contact sensor at current timestep.
    
    Args:
        env: The environment instance.
        sensor_name: Name of the contact sensor in the scene.
        threshold: Force threshold to determine contact (N).
    
    Returns:
        Contact status tensor of shape [num_envs, 2] where:
        - Column 0: Left foot contact (1.0 if in contact, 0.0 otherwise)
        - Column 1: Right foot contact (1.0 if in contact, 0.0 otherwise)
    """
    # Get contact sensor
    contact_sensor = env.scene.sensors[sensor_name]
    
    # Get current contact forces (no history needed)
    net_contact_forces = contact_sensor.data.net_forces_w  # [num_envs, num_bodies, 3]
    
    # Calculate force magnitudes
    force_magnitudes = torch.norm(net_contact_forces, dim=-1)  # [num_envs, num_bodies]
    
    # Get body names from the sensor (not from cfg)
    body_names = contact_sensor.body_names
    
    left_foot_idx = None
    right_foot_idx = None
    
    # Find body indices for ankle roll links
    for i, body_name in enumerate(body_names):
        if "left_ankle_roll_link" in body_name:
            left_foot_idx = i
        elif "right_ankle_roll_link" in body_name:
            right_foot_idx = i
    
    # If we can't find specific indices, try pattern matching
    if left_foot_idx is None or right_foot_idx is None:
        # Try to find indices by pattern matching with the actual body IDs
        import re
        for i, body_name in enumerate(body_names):
            if re.match(r".*left.*ankle_roll_link", body_name):
                left_foot_idx = i
            elif re.match(r".*right.*ankle_roll_link", body_name):
                right_foot_idx = i
    
    # Create contact tensor
    num_envs = env.num_envs
    foot_contacts = torch.zeros(num_envs, 2, device=env.device)
    
    if left_foot_idx is not None:
        foot_contacts[:, 0] = (force_magnitudes[:, left_foot_idx] > threshold).float()
    
    if right_foot_idx is not None:
        foot_contacts[:, 1] = (force_magnitudes[:, right_foot_idx] > threshold).float()
    
    return foot_contacts
