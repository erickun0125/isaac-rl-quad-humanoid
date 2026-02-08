"""GO2 velocity MDP terms: rewards, curriculums, and events."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# --- Rewards ---


def nominal_joint_pos_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float = 1.0,
    velocity_threshold: float = 0.01,
    hip_weight: float = 2.0,
    other_joint_weight: float = 1.0,
    std: float = 2.0,
) -> torch.Tensor:
    """Reward for maintaining nominal joint positions, stronger when stationary.

    Uses a Gaussian reward based on weighted joint position errors.
    Hip joints receive higher weight. The reward is scaled up when
    the velocity command is near zero (stand-still mode).
    """
    asset: Articulation = env.scene[asset_cfg.name]

    cmd = env.command_manager.get_command("base_velocity")
    cmd_magnitude = torch.norm(cmd[:, :3], dim=1)
    cmd_is_static = cmd_magnitude <= velocity_threshold

    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_squared_error = joint_error ** 2

    joint_weights = torch.ones(12, device=env.device)
    joint_weights[0:4] = hip_weight
    joint_weights[4:12] = other_joint_weight

    weighted_squared_error = joint_squared_error * joint_weights.unsqueeze(0)
    total_weighted_error = torch.sum(weighted_squared_error, dim=1)

    base_reward = torch.exp(-0.5 * (torch.sqrt(total_weighted_error) / std) ** 2)

    reward = torch.where(cmd_is_static, stand_still_scale * base_reward, base_reward)
    return reward


def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward swinging feet for clearing a specified height off the ground."""
    asset = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output."""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)


# --- Curriculums ---


def _get_num_steps_per_env(env: ManagerBasedRLEnv) -> int:
    """Dynamically get num_steps_per_env from the environment or runner config."""
    if hasattr(env, '_runner_cfg') and hasattr(env._runner_cfg, 'num_steps_per_env'):
        return env._runner_cfg.num_steps_per_env
    else:
        return 24


def modify_physics_material_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
    warmup_steps: int = 1000,
    initial_friction_range: tuple[float, float] = (0.8, 0.8),
    final_friction_range: tuple[float, float] = (0.2, 2.0),
    initial_restitution_range: tuple[float, float] = (0.0, 0.0),
    final_restitution_range: tuple[float, float] = (0.0, 0.6),
) -> dict[str, float]:
    """Progressively expand friction and restitution ranges."""
    num_steps_per_env = _get_num_steps_per_env(env)
    current_iteration = env.common_step_counter // num_steps_per_env

    if current_iteration < warmup_steps:
        return {"progress": 0.0}

    curriculum_start = warmup_steps
    curriculum_end = warmup_steps + num_steps

    if current_iteration >= curriculum_end:
        progress = 1.0
        friction_range = final_friction_range
        restitution_range = final_restitution_range
    else:
        progress = (current_iteration - curriculum_start) / num_steps
        friction_low = initial_friction_range[0] + progress * (final_friction_range[0] - initial_friction_range[0])
        friction_high = initial_friction_range[1] + progress * (final_friction_range[1] - initial_friction_range[1])
        friction_range = (friction_low, friction_high)

        restitution_low = initial_restitution_range[0] + progress * (final_restitution_range[0] - initial_restitution_range[0])
        restitution_high = initial_restitution_range[1] + progress * (final_restitution_range[1] - initial_restitution_range[1])
        restitution_range = (restitution_low, restitution_high)

    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["static_friction_range"] = friction_range
    term_cfg.params["dynamic_friction_range"] = friction_range
    term_cfg.params["restitution_range"] = restitution_range
    env.event_manager.set_term_cfg(term_name, term_cfg)

    return {"progress": float(progress)}


def modify_external_forces_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
    warmup_steps: int = 1000,
    initial_force_range: tuple[float, float] = (0.0, 0.0),
    final_force_range: tuple[float, float] = (-50.0, 50.0),
    initial_torque_range: tuple[float, float] = (0.0, 0.0),
    final_torque_range: tuple[float, float] = (-10.0, 10.0),
) -> dict[str, float]:
    """Progressively increase external force and torque ranges."""
    num_steps_per_env = _get_num_steps_per_env(env)
    current_iteration = env.common_step_counter // num_steps_per_env

    if current_iteration < warmup_steps:
        return {"progress": 0.0}

    curriculum_start = warmup_steps
    curriculum_end = warmup_steps + num_steps

    if current_iteration >= curriculum_end:
        progress = 1.0
        force_range = final_force_range
        torque_range = final_torque_range
    else:
        progress = (current_iteration - curriculum_start) / num_steps
        force_low = initial_force_range[0] + progress * (final_force_range[0] - initial_force_range[0])
        force_high = initial_force_range[1] + progress * (final_force_range[1] - initial_force_range[1])
        force_range = (force_low, force_high)

        torque_low = initial_torque_range[0] + progress * (final_torque_range[0] - initial_torque_range[0])
        torque_high = initial_torque_range[1] + progress * (final_torque_range[1] - initial_torque_range[1])
        torque_range = (torque_low, torque_high)

    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["force_range"] = force_range
    term_cfg.params["torque_range"] = torque_range
    env.event_manager.set_term_cfg(term_name, term_cfg)

    return {"progress": float(progress)}


def modify_push_robot_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
    warmup_steps: int = 1000,
    initial_velocity_range: dict[str, tuple[float, float]] = {"x": (0.0, 0.0), "y": (0.0, 0.0)},
    final_velocity_range: dict[str, tuple[float, float]] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
    initial_interval: tuple[float, float] = (15.0, 20.0),
    final_interval: tuple[float, float] = (5.0, 10.0),
) -> dict[str, float]:
    """Progressively increase push robot intensity and frequency."""
    num_steps_per_env = _get_num_steps_per_env(env)
    current_iteration = env.common_step_counter // num_steps_per_env

    if current_iteration < warmup_steps:
        return {"progress": 0.0}

    curriculum_start = warmup_steps
    curriculum_end = warmup_steps + num_steps

    if current_iteration >= curriculum_end:
        progress = 1.0
        interval_range = final_interval
        velocity_range_dict = final_velocity_range
    else:
        progress = (current_iteration - curriculum_start) / num_steps

        interval_low = initial_interval[0] + progress * (final_interval[0] - initial_interval[0])
        interval_high = initial_interval[1] + progress * (final_interval[1] - initial_interval[1])
        interval_range = (interval_low, interval_high)

        x_low = initial_velocity_range["x"][0] + progress * (final_velocity_range["x"][0] - initial_velocity_range["x"][0])
        x_high = initial_velocity_range["x"][1] + progress * (final_velocity_range["x"][1] - initial_velocity_range["x"][1])
        y_low = initial_velocity_range["y"][0] + progress * (final_velocity_range["y"][0] - initial_velocity_range["y"][0])
        y_high = initial_velocity_range["y"][1] + progress * (final_velocity_range["y"][1] - initial_velocity_range["y"][1])
        velocity_range_dict = {"x": (x_low, x_high), "y": (y_low, y_high)}

    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.interval_range_s = interval_range
    term_cfg.params["velocity_range"] = velocity_range_dict
    env.event_manager.set_term_cfg(term_name, term_cfg)

    return {"progress": float(progress)}


def modify_velocity_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    num_steps: int,
    warmup_steps: int = 1000,
    initial_lin_vel_x: tuple[float, float] = (-1.0, 1.0),
    final_lin_vel_x: tuple[float, float] = (-2.5, 2.5),
    initial_lin_vel_y: tuple[float, float] = (-1.0, 1.0),
    final_lin_vel_y: tuple[float, float] = (-1.5, 1.5),
    initial_ang_vel_z: tuple[float, float] = (-1.0, 1.0),
    final_ang_vel_z: tuple[float, float] = (-2.0, 2.0),
) -> dict[str, float]:
    """Progressively expand velocity command ranges."""
    num_steps_per_env = _get_num_steps_per_env(env)
    current_iteration = env.common_step_counter // num_steps_per_env

    if current_iteration < warmup_steps:
        return {"progress": 0.0}

    curriculum_start = warmup_steps
    curriculum_end = warmup_steps + num_steps

    if current_iteration >= curriculum_end:
        progress = 1.0
        lin_vel_x = final_lin_vel_x
        lin_vel_y = final_lin_vel_y
        ang_vel_z = final_ang_vel_z
    else:
        progress = (current_iteration - curriculum_start) / num_steps

        lin_vel_x_low = initial_lin_vel_x[0] + progress * (final_lin_vel_x[0] - initial_lin_vel_x[0])
        lin_vel_x_high = initial_lin_vel_x[1] + progress * (final_lin_vel_x[1] - initial_lin_vel_x[1])
        lin_vel_x = (lin_vel_x_low, lin_vel_x_high)

        lin_vel_y_low = initial_lin_vel_y[0] + progress * (final_lin_vel_y[0] - initial_lin_vel_y[0])
        lin_vel_y_high = initial_lin_vel_y[1] + progress * (final_lin_vel_y[1] - initial_lin_vel_y[1])
        lin_vel_y = (lin_vel_y_low, lin_vel_y_high)

        ang_vel_z_low = initial_ang_vel_z[0] + progress * (final_ang_vel_z[0] - initial_ang_vel_z[0])
        ang_vel_z_high = initial_ang_vel_z[1] + progress * (final_ang_vel_z[1] - initial_ang_vel_z[1])
        ang_vel_z = (ang_vel_z_low, ang_vel_z_high)

    cmd_term = env.command_manager.get_term(command_name)
    cmd_term.cfg.ranges.lin_vel_x = lin_vel_x
    cmd_term.cfg.ranges.lin_vel_y = lin_vel_y
    cmd_term.cfg.ranges.ang_vel_z = ang_vel_z

    return {"progress": float(progress)}


def modify_mass_randomization_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
    warmup_steps: int = 1000,
    initial_mass_range: tuple[float, float] = (-1.0, 3.0),
    final_mass_range: tuple[float, float] = (-8.0, 8.0),
) -> dict[str, float]:
    """Progressively expand mass randomization range."""
    num_steps_per_env = _get_num_steps_per_env(env)
    current_iteration = env.common_step_counter // num_steps_per_env

    if current_iteration < warmup_steps:
        return {"progress": 0.0}

    curriculum_start = warmup_steps
    curriculum_end = warmup_steps + num_steps

    if current_iteration >= curriculum_end:
        progress = 1.0
        mass_range = final_mass_range
    else:
        progress = (current_iteration - curriculum_start) / num_steps
        mass_low = initial_mass_range[0] + progress * (final_mass_range[0] - initial_mass_range[0])
        mass_high = initial_mass_range[1] + progress * (final_mass_range[1] - initial_mass_range[1])
        mass_range = (mass_low, mass_high)

    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["mass_distribution_params"] = mass_range
    env.event_manager.set_term_cfg(term_name, term_cfg)

    return {"progress": float(progress)}


def modify_reward_weight_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    num_steps: int,
    warmup_steps: int = 1000,
    initial_weight: float = 1.0,
    final_weight: float = 0.0,
    decay_type: str = "linear",
) -> dict[str, float]:
    """Progressively adjust reward term weight over training.

    Supports linear, exponential, and cosine decay types.
    Useful for phasing out rewards that help early training but hinder later stages.
    """
    num_steps_per_env = _get_num_steps_per_env(env)
    current_iteration = env.common_step_counter // num_steps_per_env

    if current_iteration < warmup_steps:
        current_weight = initial_weight
        progress = 0.0
        return {"progress": float(progress), "current_weight": float(current_weight)}

    curriculum_start = warmup_steps
    curriculum_end = warmup_steps + num_steps

    if current_iteration >= curriculum_end:
        current_weight = final_weight
        progress = 1.0
    else:
        raw_progress = (current_iteration - curriculum_start) / num_steps

        if decay_type == "linear":
            progress = raw_progress
        elif decay_type == "exponential":
            progress = 1.0 - math.exp(-3.0 * raw_progress)
        elif decay_type == "cosine":
            progress = 0.5 * (1.0 + math.cos(math.pi * (1.0 - raw_progress)))
        else:
            progress = raw_progress

        current_weight = initial_weight + progress * (final_weight - initial_weight)

    try:
        term_idx = env.reward_manager._term_names.index(reward_term_name)
        env.reward_manager._term_cfgs[term_idx].weight = current_weight
    except (ValueError, AttributeError) as e:
        print(f"Warning: Could not update reward term '{reward_term_name}': {e}")

    return {"progress": float(progress), "current_weight": float(current_weight)}


# --- Events ---


def selective_external_force_torque(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stable_env_ratio: float = 0.2,
):
    """Apply external forces to only a fraction of environments.

    A ratio of environments (determined by stable_env_ratio) are kept force-free
    for stable learning. The front env IDs are designated as stable.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    num_stable_envs = int(len(env_ids) * stable_env_ratio)
    if num_stable_envs > 0:
        stable_env_mask = env_ids < num_stable_envs
        affected_env_ids = env_ids[~stable_env_mask]
    else:
        affected_env_ids = env_ids

    if len(affected_env_ids) > 0:
        mdp.apply_external_force_torque(env, affected_env_ids, force_range, torque_range, asset_cfg)


def selective_push_by_setting_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stable_env_ratio: float = 0.2,
):
    """Push robot by setting velocity on only a fraction of environments.

    A ratio of environments (determined by stable_env_ratio) are kept push-free
    for stable learning. The front env IDs are designated as stable.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)

    num_stable_envs = int(len(env_ids) * stable_env_ratio)
    if num_stable_envs > 0:
        stable_env_mask = env_ids < num_stable_envs
        affected_env_ids = env_ids[~stable_env_mask]
    else:
        affected_env_ids = env_ids

    if len(affected_env_ids) > 0:
        mdp.push_by_setting_velocity(env, affected_env_ids, velocity_range, asset_cfg)
