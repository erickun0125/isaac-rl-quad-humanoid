# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recovery-specific observation terms for the MDP."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def episode_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Get the normalized episode progress (t/T) for each environment.
    
    Args:
        env: The environment instance
        
    Returns:
        Episode progress tensor [num_envs, 1] with values from 0 to 1
    """
    current_step = env.episode_length_buf.float()  # [num_envs] - current episode step
    max_steps = float(env.max_episode_length)      # maximum episode length
    progress_ratio = current_step / max_steps      # [num_envs] - normalized progress (0-1)
    return progress_ratio.unsqueeze(-1)  # [num_envs, 1] - add dimension for concatenation
