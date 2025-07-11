# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Curriculum terms for Go2 locomotion training with progressive difficulty."""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Dynamic configuration - agent에서 가져오도록 개선
def _get_num_steps_per_env(env: ManagerBasedRLEnv) -> int:
    """환경에서 동적으로 num_steps_per_env 값을 가져오는 함수."""
    # Try to get from runner config if available
    if hasattr(env, '_runner_cfg') and hasattr(env._runner_cfg, 'num_steps_per_env'):
        return env._runner_cfg.num_steps_per_env
    else:
        # Fallback to default value
        return 24

def modify_physics_material_curriculum(
    env: ManagerBasedRLEnv, 
    env_ids: Sequence[int], 
    term_name: str, 
    num_steps: int,
    warmup_steps: int = 1000,  # 1500 -> 1000으로 단축 (더 빠른 curriculum 시작)
    initial_friction_range: tuple[float, float] = (0.8, 0.8),  # 실제 Go2 기본값
    final_friction_range: tuple[float, float] = (0.2, 2.0),
    initial_restitution_range: tuple[float, float] = (0.0, 0.0),  # 실제 Go2 기본값
    final_restitution_range: tuple[float, float] = (0.0, 0.6),
) -> dict[str, float]:
    """Physics material curriculum: 마찰력과 반발계수를 점진적으로 확장."""
    
    # Dynamic num_steps_per_env 사용
    num_steps_per_env = _get_num_steps_per_env(env)
    
    # Training iteration 계산: common_step_counter를 training iteration으로 변환
    current_iteration = env.common_step_counter // num_steps_per_env
    
    # Warmup 기간 동안은 기본값 유지
    if current_iteration < warmup_steps:
        return {
            "progress": 0.0,
        }
    
    # Curriculum 진행도 계산 (0.0 ~ 1.0)
    curriculum_start = warmup_steps
    curriculum_end = warmup_steps + num_steps
    
    if current_iteration >= curriculum_end:
        # Curriculum 완료 - 최종값 사용
        progress = 1.0
        friction_range = final_friction_range
        restitution_range = final_restitution_range
    else:
        # Curriculum 진행 중 - 선형 보간
        progress = (current_iteration - curriculum_start) / num_steps
        friction_low = initial_friction_range[0] + progress * (final_friction_range[0] - initial_friction_range[0])
        friction_high = initial_friction_range[1] + progress * (final_friction_range[1] - initial_friction_range[1])
        friction_range = (friction_low, friction_high)
        
        restitution_low = initial_restitution_range[0] + progress * (final_restitution_range[0] - initial_restitution_range[0])
        restitution_high = initial_restitution_range[1] + progress * (final_restitution_range[1] - initial_restitution_range[1])
        restitution_range = (restitution_low, restitution_high)
    
    # Material 속성 업데이트
    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["static_friction_range"] = friction_range
    term_cfg.params["dynamic_friction_range"] = friction_range
    term_cfg.params["restitution_range"] = restitution_range
    env.event_manager.set_term_cfg(term_name, term_cfg)
    
    return {
        "progress": float(progress),
    }


def modify_external_forces_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int], 
    term_name: str,
    num_steps: int,
    warmup_steps: int = 1000,  # 1500 -> 1000으로 단축
    initial_force_range: tuple[float, float] = (0.0, 0.0),  # 실제 Go2 기본값
    final_force_range: tuple[float, float] = (-50.0, 50.0),
    initial_torque_range: tuple[float, float] = (0.0, 0.0),  # 실제 Go2 기본값  
    final_torque_range: tuple[float, float] = (-10.0, 10.0),
) -> dict[str, float]:
    """External forces curriculum: 외부 힘과 토크를 점진적으로 증가."""
    
    # Dynamic num_steps_per_env 사용
    num_steps_per_env = _get_num_steps_per_env(env)
    
    # Training iteration 계산
    current_iteration = env.common_step_counter // num_steps_per_env
    
    # Warmup 기간 동안은 기본값 유지
    if current_iteration < warmup_steps:
        return {
            "progress": 0.0,
        }
    
    # Curriculum 진행도 계산
    curriculum_start = warmup_steps
    curriculum_end = warmup_steps + num_steps
    
    if current_iteration >= curriculum_end:
        # Curriculum 완료
        progress = 1.0
        force_range = final_force_range
        torque_range = final_torque_range
    else:
        # Curriculum 진행 중
        progress = (current_iteration - curriculum_start) / num_steps
        force_low = initial_force_range[0] + progress * (final_force_range[0] - initial_force_range[0])
        force_high = initial_force_range[1] + progress * (final_force_range[1] - initial_force_range[1])
        force_range = (force_low, force_high)
        
        torque_low = initial_torque_range[0] + progress * (final_torque_range[0] - initial_torque_range[0])
        torque_high = initial_torque_range[1] + progress * (final_torque_range[1] - initial_torque_range[1])
        torque_range = (torque_low, torque_high)
    
    # External forces 업데이트  
    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["force_range"] = force_range  # tuple (min, max) 형태로 설정
    term_cfg.params["torque_range"] = torque_range  # tuple (min, max) 형태로 설정
    env.event_manager.set_term_cfg(term_name, term_cfg)
    
    return {
        "progress": float(progress),
    }


def modify_push_robot_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str, 
    num_steps: int,
    warmup_steps: int = 1000,  # 1500 -> 1000으로 단축
    initial_velocity_range: dict[str, tuple[float, float]] = {"x": (0.0, 0.0), "y": (0.0, 0.0)},  # Go2에서 push_robot=None이므로 0부터 시작
    final_velocity_range: dict[str, tuple[float, float]] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
    initial_interval: tuple[float, float] = (15.0, 20.0),  
    final_interval: tuple[float, float] = (5.0, 10.0),
) -> dict[str, float]:
    """Push robot curriculum: 로봇 푸시 강도와 빈도를 점진적으로 증가."""
    
    # Dynamic num_steps_per_env 사용
    num_steps_per_env = _get_num_steps_per_env(env)
    
    # Training iteration 계산
    current_iteration = env.common_step_counter // num_steps_per_env
    
    # Warmup 기간 동안은 기본값 유지
    if current_iteration < warmup_steps:
        return {
            "progress": 0.0,
        }
    
    # Curriculum 진행도 계산
    curriculum_start = warmup_steps
    curriculum_end = warmup_steps + num_steps
    
    if current_iteration >= curriculum_end:
        # Curriculum 완료
        progress = 1.0
        interval_range = final_interval
        velocity_range_dict = final_velocity_range
    else:
        # Curriculum 진행 중
        progress = (current_iteration - curriculum_start) / num_steps
        
        # 간격 보간 (더 자주 푸시)
        interval_low = initial_interval[0] + progress * (final_interval[0] - initial_interval[0])
        interval_high = initial_interval[1] + progress * (final_interval[1] - initial_interval[1])
        interval_range = (interval_low, interval_high)
        
        # 속도 범위 보간
        x_low = initial_velocity_range["x"][0] + progress * (final_velocity_range["x"][0] - initial_velocity_range["x"][0])
        x_high = initial_velocity_range["x"][1] + progress * (final_velocity_range["x"][1] - initial_velocity_range["x"][1])
        y_low = initial_velocity_range["y"][0] + progress * (final_velocity_range["y"][0] - initial_velocity_range["y"][0])
        y_high = initial_velocity_range["y"][1] + progress * (final_velocity_range["y"][1] - initial_velocity_range["y"][1])
        velocity_range_dict = {"x": (x_low, x_high), "y": (y_low, y_high)}
    
    # Push 설정 업데이트
    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.interval_range_s = interval_range
    term_cfg.params["velocity_range"] = velocity_range_dict
    env.event_manager.set_term_cfg(term_name, term_cfg)
    
    return {
        "progress": float(progress),
    }


def modify_velocity_command_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str,
    num_steps: int,
    warmup_steps: int = 1000,  # 1500 -> 1000으로 단축
    initial_lin_vel_x: tuple[float, float] = (-1.0, 1.0),  # 실제 Go2 기본값
    final_lin_vel_x: tuple[float, float] = (-2.5, 2.5),
    initial_lin_vel_y: tuple[float, float] = (-1.0, 1.0),  # 실제 Go2 기본값
    final_lin_vel_y: tuple[float, float] = (-1.5, 1.5), 
    initial_ang_vel_z: tuple[float, float] = (-1.0, 1.0),  # 실제 Go2 기본값
    final_ang_vel_z: tuple[float, float] = (-2.0, 2.0),
) -> dict[str, float]:
    """Velocity command curriculum: 속도 명령 범위를 점진적으로 확장."""
    
    # Dynamic num_steps_per_env 사용
    num_steps_per_env = _get_num_steps_per_env(env)
    
    # Training iteration 계산
    current_iteration = env.common_step_counter // num_steps_per_env
    
    # Warmup 기간 동안은 기본값 유지
    if current_iteration < warmup_steps:
        return {
            "progress": 0.0,
        }
    
    # Curriculum 진행도 계산
    curriculum_start = warmup_steps
    curriculum_end = warmup_steps + num_steps
    
    if current_iteration >= curriculum_end:
        # Curriculum 완료
        progress = 1.0
        lin_vel_x = final_lin_vel_x
        lin_vel_y = final_lin_vel_y
        ang_vel_z = final_ang_vel_z
    else:
        # Curriculum 진행 중
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
    
    # Command 범위 업데이트
    cmd_term = env.command_manager.get_term(command_name)
    cmd_term.cfg.ranges.lin_vel_x = lin_vel_x
    cmd_term.cfg.ranges.lin_vel_y = lin_vel_y
    cmd_term.cfg.ranges.ang_vel_z = ang_vel_z
    
    return {
        "progress": float(progress),
    }


def modify_mass_randomization_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    num_steps: int,
    warmup_steps: int = 1000,  # 1500 -> 1000으로 단축
    initial_mass_range: tuple[float, float] = (-1.0, 3.0),  # 실제 Go2 기본값 (수정됨)
    final_mass_range: tuple[float, float] = (-8.0, 8.0),
) -> dict[str, float]:
    """Mass randomization curriculum: 질량 변화 범위를 점진적으로 확장."""
    
    # Dynamic num_steps_per_env 사용
    num_steps_per_env = _get_num_steps_per_env(env)
    
    # Training iteration 계산
    current_iteration = env.common_step_counter // num_steps_per_env
    
    # Warmup 기간 동안은 기본값 유지
    if current_iteration < warmup_steps:
        return {
            "progress": 0.0,
        }
    
    # Curriculum 진행도 계산
    curriculum_start = warmup_steps
    curriculum_end = warmup_steps + num_steps
    
    if current_iteration >= curriculum_end:
        # Curriculum 완료
        progress = 1.0
        mass_range = final_mass_range
    else:
        # Curriculum 진행 중
        progress = (current_iteration - curriculum_start) / num_steps
        mass_low = initial_mass_range[0] + progress * (final_mass_range[0] - initial_mass_range[0])
        mass_high = initial_mass_range[1] + progress * (final_mass_range[1] - initial_mass_range[1])
        mass_range = (mass_low, mass_high)
    
    # Mass distribution 업데이트
    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["mass_distribution_params"] = mass_range
    env.event_manager.set_term_cfg(term_name, term_cfg)
    
    return {
        "progress": float(progress),
    } 


def modify_reward_weight_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    num_steps: int,
    warmup_steps: int = 1000,
    initial_weight: float = 1.0,
    final_weight: float = 0.0,
    decay_type: str = "linear",  # "linear", "exponential", "cosine"
) -> dict[str, float]:
    """Reward weight curriculum: 특정 reward term의 weight를 점진적으로 조절.
    
    warmup 기간 이후부터 reward weight를 점진적으로 변경합니다.
    주로 초기 학습에 도움이 되지만 후기에는 방해가 되는 reward들에 사용합니다.
    
    Args:
        env: RL 환경
        env_ids: 환경 ID들 (사용되지 않음, 호환성을 위해 유지)
        reward_term_name: 조절할 reward term 이름
        num_steps: Curriculum 지속 기간 (training iterations)
        warmup_steps: Warmup 기간 (training iterations)
        initial_weight: 초기 weight 값
        final_weight: 최종 weight 값
        decay_type: 감소 방식 ("linear", "exponential", "cosine")
    
    Returns:
        Curriculum 상태 정보
    """
    # Dynamic num_steps_per_env 사용
    num_steps_per_env = _get_num_steps_per_env(env)
    
    # Training iteration 계산
    current_iteration = env.common_step_counter // num_steps_per_env
    
    # Warmup 기간 동안은 초기 weight 유지
    if current_iteration < warmup_steps:
        current_weight = initial_weight
        progress = 0.0
        return {
            "progress": float(progress),
            "current_weight": float(current_weight),
        }
    else:
        # Curriculum 진행도 계산
        curriculum_start = warmup_steps
        curriculum_end = warmup_steps + num_steps
        
        if current_iteration >= curriculum_end:
            # Curriculum 완료 - 최종 weight 사용
            current_weight = final_weight
            progress = 1.0
        else:
            # Curriculum 진행 중 - 점진적 변화
            raw_progress = (current_iteration - curriculum_start) / num_steps
            
            if decay_type == "linear":
                progress = raw_progress
            elif decay_type == "exponential":
                # 지수적 감소 (더 빠른 초기 감소)
                progress = 1.0 - math.exp(-3.0 * raw_progress)
            elif decay_type == "cosine":
                # 코사인 감소 (부드러운 곡선)
                progress = 0.5 * (1.0 + math.cos(math.pi * (1.0 - raw_progress)))
            else:
                progress = raw_progress  # 기본값은 linear
            
            current_weight = initial_weight + progress * (final_weight - initial_weight)
    
    # Reward manager에서 해당 term의 weight 업데이트
    try:
        # Get the reward term configuration
        term_idx = env.reward_manager._term_names.index(reward_term_name)
        env.reward_manager._term_cfgs[term_idx].weight = current_weight
    except (ValueError, AttributeError) as e:
        # Term not found or reward manager access failed
        print(f"Warning: Could not update reward term '{reward_term_name}': {e}")
    
    return {
        "progress": float(progress),
        "current_weight": float(current_weight),
    } 