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



def nominal_joint_pos_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float = 1.0,
    velocity_threshold: float = 0.01,
    hip_weight: float = 2.0,
    other_joint_weight: float = 1.0,
    std: float = 2.0,
) -> torch.Tensor:

    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # Velocity command 확인 - threshold 기반으로 static 여부 판단
    cmd = env.command_manager.get_command("base_velocity")
    cmd_magnitude = torch.norm(cmd[:, :3], dim=1)  # x, y, yaw의 magnitude
    cmd_is_static = cmd_magnitude <= velocity_threshold
    
    # Joint position error 계산 (Squared error)
    joint_error = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_squared_error = joint_error ** 2
    
    # Go2 로봇의 joint 구조에 맞는 가중치 적용
    # Go2 joints: [FL_hip(0), FR_hip(1), RL_hip(2), RR_hip(3),
    #              FL_thigh(4), FR_thigh(5), RL_thigh(6), RR_thigh(7),
    #              FL_calf(8), FR_calf(9), RL_calf(10), RR_calf(11)]
    
    # 가중치 텐서 생성
    joint_weights = torch.ones(12, device=env.device)
    
    # Hip joints (0-3)에 더 큰 가중치 적용
    joint_weights[0:4] = hip_weight  # FL_hip, FR_hip, RL_hip, RR_hip
    
    # 나머지 joints (4-11)에는 기본 가중치
    joint_weights[4:12] = other_joint_weight  # Thigh and Calf joints
    
    # 가중치를 적용한 weighted squared error 계산
    weighted_squared_error = joint_squared_error * joint_weights.unsqueeze(0)
    total_weighted_error = torch.sum(weighted_squared_error, dim=1)
    
    # Reward 계산: Gaussian (bell-shaped) function 사용
    # total_weighted_error가 0일 때 exp(0) = 1 (최대 reward)
    # std가 클수록 더 관대한 reward, 작을수록 더 엄격한 reward
    base_reward = torch.exp(-0.5 * (torch.sqrt(total_weighted_error) / std) ** 2)
    
    # Static 상태일 때만 stand_still_scale 배만큼 강화, 그 외에는 낮은 reward
    reward = torch.where(
        cmd_is_static, 
        stand_still_scale * base_reward,
        base_reward  # 움직일 때는 30%만 적용
    )
    
    return reward

def foot_clearance_reward(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, target_height: float, std: float, tanh_mult: float
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = foot_z_target_error * foot_velocity_tanh
    return torch.exp(-torch.sum(reward, dim=1) / std)


def action_smoothness_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize large instantaneous changes in the network action output"""
    return torch.linalg.norm((env.action_manager.action - env.action_manager.prev_action), dim=1)