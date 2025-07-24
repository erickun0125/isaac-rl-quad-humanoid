# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recovery-specific event terms for the MDP."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_with_physics_simulation(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    roll_range: tuple[float, float],
    pitch_range: tuple[float, float],
    height_range: tuple[float, float],
    simulation_time: float = 1.0,
    joint_pos_noise_range: tuple[float, float] = (-0.5, 0.5),
    joint_vel_range: tuple[float, float] = (-3.0, 3.0),
    root_lin_vel_range: tuple[float, float] = (-0.5, 0.5),
    root_ang_vel_range: tuple[float, float] = (-1.5, 1.5),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset function that drops the robot in random configuration and runs physics simulation for T seconds.
    
    This function:
    1. Places the robot in a random fallen configuration with diverse joint positions
    2. Sets joint gains to zero (free joints) and runs physics simulation for T seconds  
    3. Restores joint gains after simulation for policy control
    
    Args:
        env: The environment instance
        env_ids: Environment IDs to reset
        pose_range: Range for sampling the asset root pose. Expected keys: "x", "y", "yaw"
        roll_range: Range for sampling the roll angle (robot can be on its side)
        pitch_range: Range for sampling the pitch angle (robot can be tilted)
        height_range: Range for sampling the height above ground (for dropping)
        simulation_time: Time in seconds to run physics simulation before policy starts
        joint_pos_noise_range: Range for joint position noise (min, max) in radians
        joint_vel_range: Range for joint velocity initialization (min, max) in rad/s
        root_lin_vel_range: Range for root linear velocity initialization (min, max) in m/s
        root_ang_vel_range: Range for root angular velocity initialization (min, max) in rad/s
        asset_cfg: Scene entity configuration for the asset
    """
    # Extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]

    # Store original joint gains to restore later
    original_stiffness = asset.actuators["base_legs"].stiffness.clone()
    original_damping = asset.actuators["base_legs"].damping.clone()

    # 1. Set initial random fallen configuration
    root_state = asset.data.default_root_state[env_ids].clone()

    # Sample random poses
    # -- xy-position
    range_list = [pose_range["x"], pose_range["y"]]
    for i, attr in enumerate(["x", "y"]):
        root_state[:, i] = root_state[:, i].uniform_(*range_list[i])
    
    # -- height (z-position) - drop from height for realistic falling
    root_state[:, 2] = torch.empty(len(env_ids), device=env.device).uniform_(*height_range)
    
    # -- orientation: sample random roll, pitch, and yaw for fallen states
    roll = torch.empty(len(env_ids), device=env.device).uniform_(*roll_range)
    pitch = torch.empty(len(env_ids), device=env.device).uniform_(*pitch_range)
    yaw = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range["yaw"])
    
    # Convert to quaternion
    quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    root_state[:, 3:7] = quat
    
    # Set random initial velocities for more realistic dropping using uniform distribution
    root_state[:, 7:10] = torch.empty(len(env_ids), 3, device=env.device).uniform_(*root_lin_vel_range)  # Linear velocity
    root_state[:, 10:13] = torch.empty(len(env_ids), 3, device=env.device).uniform_(*root_ang_vel_range)  # Angular velocity

    # Write initial state to simulation
    asset.write_root_state_to_sim(root_state, env_ids)

    # Set diverse random joint positions - each joint gets independent random value
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    
    # Add independent random noise to each joint using uniform distribution
    joint_noise = torch.empty_like(joint_pos).uniform_(*joint_pos_noise_range)
    joint_pos += joint_noise
    
    # Clamp joint positions to safe limits to avoid impossible configurations
    joint_pos = torch.clamp(joint_pos, 
                           asset.data.soft_joint_pos_limits[env_ids, :, 0], 
                           asset.data.soft_joint_pos_limits[env_ids, :, 1])
    
    # Set random joint velocities for more dynamic initial state using uniform distribution
    joint_vel = torch.empty_like(joint_pos).uniform_(*joint_vel_range)
    
    # Write joint state to simulation
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # 2. Set joint gains to zero for completely free joints during physics simulation
    zero_stiffness = torch.zeros_like(original_stiffness)
    zero_damping = torch.zeros_like(original_damping)
    
    # Apply zero gains to make joints completely free
    asset.actuators["base_legs"].stiffness[:] = zero_stiffness
    asset.actuators["base_legs"].damping[:] = zero_damping

    # 3. Run physics simulation for T seconds (with completely free joints)
    physics_dt = env.physics_dt
    num_steps = int(simulation_time / physics_dt)
    
    for step in range(num_steps):
        # No need to set effort targets - joints are free (zero gains)
        # This allows natural physics-based motion during falling
        
        # Write scene data to simulation
        env.scene.write_data_to_sim()
        
        # Step physics simulation (without rendering for speed)
        env.sim.step(render=False)
        
        # Update scene data
        env.scene.update(dt=physics_dt)

    # 4. Restore original joint gains for policy control
    asset.actuators["base_legs"].stiffness[:] = original_stiffness
    asset.actuators["base_legs"].damping[:] = original_damping

    # 5. Reset all joint velocities to zero for stable policy start
    asset.data.joint_vel[env_ids] = 0.0

    # Final state is now settled and ready for policy control with restored gains