# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Go2 specific reward functions for locomotion training."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def nominal_joint_pos_when_static_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float = 0.1,
) -> torch.Tensor:
    """정지 상태에서 nominal joint position 유지를 보상하는 함수.
    
    정지 조건:
    - Velocity command가 정확히 (0, 0, 0)
    
    Args:
        env: RL 환경
        asset_cfg: 로봇 asset 설정
        stand_still_scale: 정지 상태에서 적용할 scale factor
    
    Returns:
        Reward tensor (정지 상태에서만 강화된 reward)
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Velocity command 확인 - 정확히 (0, 0, 0)인지 체크
    cmd = env.command_manager.get_command("base_velocity")
    cmd_is_zero = torch.all(cmd[:, :3] == 0.0, dim=1)  # x, y, yaw가 모두 0인지
    
    # Joint position error 계산 (L1 norm)
    joint_pos_error = torch.sum(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), 
        dim=1
    )
    
    # Reward 계산: exponential을 사용해서 0~1 사이 값으로 변환
    # joint_pos_error가 0일 때 exp(0) = 1 (최대 reward)
    # joint_pos_error가 클수록 exp(-error) → 0 (최소 reward)
    base_reward = torch.exp(-joint_pos_error)
    
    # command가 (0,0,0)일 때만 stand_still_scale 배만큼 강화, 그 외에는 매우 낮은 penalty
    reward = torch.where(cmd_is_zero, stand_still_scale * base_reward, 0.01 * stand_still_scale * base_reward)
    
    return reward 