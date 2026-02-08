"""GO2 recovery MDP terms: rewards, events, observations, and curriculums."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --- Rewards ---


def target_configuration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    flat_orientation_weight: float = -1.0,
    joint_pose_weight: float = 1.0,
    final_config_weight: float = 1.0,
    foot_contact_weight: float = 1.0,
    joint_orientation_threshold: float = 0.524,
    foot_orientation_threshold_lower: float = 0.471,
    foot_orientation_threshold_upper: float = 0.942,
    final_orientation_threshold: float = 0.087,
    final_joint_threshold: float = 0.1,
    final_time_threshold: float = 0.9,
    std: float = 0.5,
    big_reward: float = 100.0,
    contact_force_threshold: float = 0.1,
) -> torch.Tensor:
    """Unified reward function combining orientation, joint pose, foot contact, and final config.

    Components:
    1. Flat orientation: Encourages upright orientation (dot_product based)
    2. Joint pose: Conditional joint pose reward when robot is upright enough
    3. Foot contact: Reward for feet touching ground when semi-upright
    4. Final configuration: Binary reward when orientation + joint conditions are met
    """
    robot: Articulation = env.scene[asset_cfg.name]

    # === Common calculations ===
    root_quat_w = robot.data.root_quat_w
    robot_z_axis = math_utils.quat_apply(root_quat_w, torch.tensor([0.0, 0.0, 1.0], device=env.device).repeat(env.num_envs, 1))
    world_z_axis = torch.tensor([0.0, 0.0, 1.0], device=env.device).expand(env.num_envs, -1)

    dot_product = torch.sum(robot_z_axis * world_z_axis, dim=1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    angle_rad = torch.acos(dot_product)

    current_joint_pos = robot.data.joint_pos
    target_values = [0.35, -0.35, 0.5, -0.5, 1.36, 1.36, 1.36, 1.36, -2.65, -2.65, -2.65, -2.65]
    target_joint_pos = torch.tensor(target_values, device=env.device).expand(env.num_envs, -1)

    joint_pos_error_abs = torch.abs(current_joint_pos - target_joint_pos)
    joint_pos_error_square = torch.square(joint_pos_error_abs)

    # === 1. Flat orientation reward ===
    flat_orientation_reward_1 = 1.0 - dot_product
    flat_orientation_reward_2 = angle_rad
    flat_orientation_reward = flat_orientation_reward_1 + 0.1 * flat_orientation_reward_2 ** 2

    # === 2. Joint pose reward ===
    is_upright_for_joint_pose = angle_rad <= joint_orientation_threshold
    joint_pos_reward_per_joint = torch.exp(-0.5 * (joint_pos_error_square / std**2))
    hip_joint_pos_reward = joint_pos_reward_per_joint[:, :4]
    non_hip_joint_pos_reward = joint_pos_reward_per_joint[:, 4:]
    joint_pos_reward_component = torch.mean(hip_joint_pos_reward, dim=1) + 2.0 * torch.mean(non_hip_joint_pos_reward, dim=1)
    joint_pose_reward = torch.where(is_upright_for_joint_pose, joint_pos_reward_component, 0.0 * joint_pos_reward_component)

    # === 3. Foot contact reward ===
    is_upright_for_foot_contact = (angle_rad >= foot_orientation_threshold_lower) & (angle_rad <= foot_orientation_threshold_upper)
    contact_sensor = env.scene["contact_forces"]
    contact_forces = contact_sensor.data.net_forces_w_history
    foot_forces = contact_forces[:, -4:, -1, :]
    foot_contact_magnitude = torch.norm(foot_forces, dim=2)
    feet_in_contact = foot_contact_magnitude > contact_force_threshold
    num_feet_in_contact = torch.sum(feet_in_contact.float(), dim=1)
    foot_contact_reward_component = num_feet_in_contact / 4.0
    foot_contact_reward = torch.where(is_upright_for_foot_contact, foot_contact_reward_component, torch.zeros_like(foot_contact_reward_component))

    # === 4. Final configuration reward ===
    current_step = env.episode_length_buf.float()
    max_steps = float(env.max_episode_length)
    progress_ratio = current_step / max_steps

    orientation_satisfied = angle_rad <= final_orientation_threshold
    hip_joint_pos_error_abs = joint_pos_error_abs[:, :4]
    hip_joint_pos_error_abs_max = torch.max(hip_joint_pos_error_abs, dim=1)[0]
    joint_satisfied = hip_joint_pos_error_abs_max <= final_joint_threshold
    time_satisfied = progress_ratio >= final_time_threshold
    all_satisfied = orientation_satisfied & joint_satisfied & time_satisfied

    final_joint_pos_reward_per_joint = torch.exp(-0.5 * (joint_pos_error_square / (std / 4.0) ** 2))
    final_joint_pos_reward_component = torch.mean(final_joint_pos_reward_per_joint, dim=1)
    final_config_reward = torch.where(all_satisfied, final_joint_pos_reward_component, torch.zeros(env.num_envs, device=env.device))

    # === Combine ===
    total_reward = (
        flat_orientation_weight * flat_orientation_reward
        + joint_pose_weight * joint_pose_reward
        + final_config_weight * final_config_reward
        + foot_contact_weight * foot_contact_reward
    )

    return total_reward


def flat_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for upright orientation (squared dot product)."""
    return target_configuration(
        env, asset_cfg,
        flat_orientation_weight=-1.0,
        joint_pose_weight=0.0,
        final_config_weight=0.0,
        foot_contact_weight=0.0,
    )


def joint_pose(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for reaching target joint pose when orientation is upright."""
    return target_configuration(
        env, asset_cfg,
        flat_orientation_weight=0.0,
        joint_pose_weight=1.0,
        final_config_weight=0.0,
        foot_contact_weight=0.0,
        joint_orientation_threshold=0.524,
        std=0.2,
    )


def foot_contact(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for feet in contact when semi-upright."""
    return target_configuration(
        env, asset_cfg,
        flat_orientation_weight=0.0,
        joint_pose_weight=0.0,
        final_config_weight=0.0,
        foot_contact_weight=1.0,
        foot_orientation_threshold_lower=0.471,
        foot_orientation_threshold_upper=0.942,
    )


def final_configuration(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Final configuration reward for successful recovery."""
    return target_configuration(
        env, asset_cfg,
        flat_orientation_weight=0.0,
        joint_pose_weight=0.0,
        final_config_weight=1.0,
        foot_contact_weight=0.0,
        final_orientation_threshold=0.16,
        final_joint_threshold=0.628,
        final_time_threshold=0.8,
        std=0.2,
        big_reward=100.0,
    )


def ang_vel_xy_threshold_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 2.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for angular velocity in xy-plane above threshold."""
    robot: Articulation = env.scene[asset_cfg.name]
    ang_vel_xy = robot.data.root_ang_vel_b[:, :2]
    ang_vel_xy_magnitude = torch.norm(ang_vel_xy, dim=1)
    excess = torch.clamp(ang_vel_xy_magnitude - threshold, min=0.0)
    return torch.square(excess)


def joint_vel_threshold_penalty(
    env: ManagerBasedRLEnv,
    threshold: float = 5.0,
    early_episode_multiplier: float = 10.0,
    progress_threshold: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalty for joint velocities above threshold."""
    robot: Articulation = env.scene[asset_cfg.name]
    joint_vel_abs = torch.abs(robot.data.joint_vel)
    excess = torch.clamp(joint_vel_abs - threshold, min=0.0)
    excess_squared = torch.square(excess)
    return torch.sum(excess_squared, dim=1)


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize feet sliding while in contact with the ground."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


# --- Events ---


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
    """Reset by dropping robot in random configuration and running physics simulation.

    Steps:
    1. Place robot in random fallen configuration with diverse joint positions
    2. Set joint gains to zero (free joints) and run physics for T seconds
    3. Restore joint gains after simulation for policy control
    """
    asset: Articulation = env.scene[asset_cfg.name]

    # Store original joint gains
    original_stiffness = asset.actuators["base_legs"].stiffness.clone()
    original_damping = asset.actuators["base_legs"].damping.clone()

    # Set initial random fallen configuration
    root_state = asset.data.default_root_state[env_ids].clone()

    range_list = [pose_range["x"], pose_range["y"]]
    for i, attr in enumerate(["x", "y"]):
        root_state[:, i] = root_state[:, i].uniform_(*range_list[i])

    root_state[:, 2] = torch.empty(len(env_ids), device=env.device).uniform_(*height_range)

    roll = torch.empty(len(env_ids), device=env.device).uniform_(*roll_range)
    pitch = torch.empty(len(env_ids), device=env.device).uniform_(*pitch_range)
    yaw = torch.empty(len(env_ids), device=env.device).uniform_(*pose_range["yaw"])
    quat = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
    root_state[:, 3:7] = quat

    root_state[:, 7:10] = torch.empty(len(env_ids), 3, device=env.device).uniform_(*root_lin_vel_range)
    root_state[:, 10:13] = torch.empty(len(env_ids), 3, device=env.device).uniform_(*root_ang_vel_range)

    asset.write_root_state_to_sim(root_state, env_ids)

    # Set diverse random joint positions
    joint_pos = asset.data.default_joint_pos[env_ids].clone()
    joint_noise = torch.empty_like(joint_pos).uniform_(*joint_pos_noise_range)
    joint_pos += joint_noise
    joint_pos = torch.clamp(
        joint_pos,
        asset.data.soft_joint_pos_limits[env_ids, :, 0],
        asset.data.soft_joint_pos_limits[env_ids, :, 1],
    )

    joint_vel = torch.empty_like(joint_pos).uniform_(*joint_vel_range)
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # Set joint gains to zero for free joints during physics simulation
    zero_stiffness = torch.zeros_like(original_stiffness)
    zero_damping = torch.zeros_like(original_damping)
    asset.actuators["base_legs"].stiffness[:] = zero_stiffness
    asset.actuators["base_legs"].damping[:] = zero_damping

    # Run physics simulation for T seconds
    physics_dt = env.physics_dt
    num_steps = int(simulation_time / physics_dt)
    for step in range(num_steps):
        env.scene.write_data_to_sim()
        env.sim.step(render=False)
        env.scene.update(dt=physics_dt)

    # Restore original joint gains
    asset.actuators["base_legs"].stiffness[:] = original_stiffness
    asset.actuators["base_legs"].damping[:] = original_damping

    # Reset joint velocities to zero for stable policy start
    asset.data.joint_vel[env_ids] = 0.0


# --- Observations ---


def episode_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get normalized episode progress (t/T) for each environment.

    Returns:
        Episode progress tensor [num_envs, 1] with values from 0 to 1.
    """
    current_step = env.episode_length_buf.float()
    max_steps = float(env.max_episode_length)
    progress_ratio = current_step / max_steps
    return progress_ratio.unsqueeze(-1)


# --- Curriculums ---


def terrain_levels_recovery(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Curriculum based on the distance the robot walks during recovery.

    Returns the distance walked by the robot in the xy-plane during the current episode.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    return distance
