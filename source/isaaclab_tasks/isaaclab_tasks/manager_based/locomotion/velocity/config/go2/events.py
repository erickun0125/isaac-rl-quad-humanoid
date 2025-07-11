# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 specific event functions for locomotion training."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def selective_external_force_torque(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    force_range: tuple[float, float],
    torque_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stable_env_ratio: float = 0.2,
):
    """External forces를 일정 비율의 환경에만 적용하는 함수.
    
    안정적인 학습을 위해 일정 비율의 로봇은 external forces의 영향을 받지 않도록 합니다.
    환경 ID 기반으로 앞쪽 환경들을 안정 환경으로 지정하여 일관성을 유지합니다.
    
    Args:
        env: RL 환경
        env_ids: 적용할 환경 ID들
        force_range: Force 적용 범위 (min, max)
        torque_range: Torque 적용 범위 (min, max)
        asset_cfg: 로봇 asset 설정
        stable_env_ratio: 안정 환경 비율 (0.0~1.0)
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # 안정 환경 비율에 따라 일부 환경 제외
    num_stable_envs = int(len(env_ids) * stable_env_ratio)
    if num_stable_envs > 0:
        # 앞쪽 환경들을 안정 환경으로 지정 (일관성 있게)
        stable_env_mask = env_ids < num_stable_envs
        affected_env_ids = env_ids[~stable_env_mask]
    else:
        affected_env_ids = env_ids
    
    # 영향받는 환경들에만 external forces 적용
    if len(affected_env_ids) > 0:
        mdp.apply_external_force_torque(env, affected_env_ids, force_range, torque_range, asset_cfg)


def selective_push_by_setting_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stable_env_ratio: float = 0.2,
):
    """Push robot을 일정 비율의 환경에만 적용하는 함수.
    
    안정적인 학습을 위해 일정 비율의 로봇은 push robot의 영향을 받지 않도록 합니다.
    환경 ID 기반으로 앞쪽 환경들을 안정 환경으로 지정하여 일관성을 유지합니다.
    
    Args:
        env: RL 환경
        env_ids: 적용할 환경 ID들
        velocity_range: Velocity 적용 범위 (x, y, z, roll, pitch, yaw)
        asset_cfg: 로봇 asset 설정
        stable_env_ratio: 안정 환경 비율 (0.0~1.0)
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    
    # 안정 환경 비율에 따라 일부 환경 제외
    num_stable_envs = int(len(env_ids) * stable_env_ratio)
    if num_stable_envs > 0:
        # 앞쪽 환경들을 안정 환경으로 지정 (일관성 있게)
        stable_env_mask = env_ids < num_stable_envs
        affected_env_ids = env_ids[~stable_env_mask]
    else:
        affected_env_ids = env_ids
    
    # 영향받는 환경들에만 push robot 적용
    if len(affected_env_ids) > 0:
        mdp.push_by_setting_velocity(env, affected_env_ids, velocity_range, asset_cfg)
