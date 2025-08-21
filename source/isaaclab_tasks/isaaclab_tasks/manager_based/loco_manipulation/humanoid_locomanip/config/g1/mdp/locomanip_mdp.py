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
    min_weight: float = -0.01,
    reward_threshold: float = 0.7,
    weight_decay_step: float = 0.01,
) -> torch.Tensor:
    """Curriculum to decrease joint deviation penalty weights based on velocity tracking performance.
    
    Args:
        env: The environment instance.
        env_ids: Environment IDs to apply curriculum to.
        reward_term_names: List of joint deviation reward term names to modify.
        tracking_reward_name: Name of the tracking reward term to monitor.
        min_weight: Minimum weight for joint deviation penalties (negative value).
        reward_threshold: Reward threshold as ratio of reward weight to trigger curriculum.
        weight_decay_step: Step size for decreasing penalty weights.
    
    Returns:
        Current joint deviation weight.
    """
    if reward_term_names is None:
        reward_term_names = ["joint_deviation_hip", "joint_deviation_arms", "joint_deviation_torso"]
    tracking_reward_term = env.reward_manager.get_term_cfg(tracking_reward_name)
    tracking_reward = torch.mean(env.reward_manager._episode_sums[tracking_reward_name][env_ids]) / env.max_episode_length_s

    # Update curriculum every episode
    if env.common_step_counter % env.max_episode_length == 0:
        if tracking_reward > tracking_reward_term.weight * reward_threshold:
            # Reduce joint deviation penalty weights (make them less negative)
            for term_name in reward_term_names:
                if term_name in env.reward_manager._term_cfgs:
                    current_weight = env.reward_manager._term_cfgs[term_name].weight
                    new_weight = min(current_weight + weight_decay_step, min_weight)
                    env.reward_manager._term_cfgs[term_name].weight = new_weight

    # Return current weight of first joint deviation term
    if reward_term_names[0] in env.reward_manager._term_cfgs:
        return torch.tensor(env.reward_manager._term_cfgs[reward_term_names[0]].weight, device=env.device)
    else:
        return torch.tensor(min_weight, device=env.device)


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
