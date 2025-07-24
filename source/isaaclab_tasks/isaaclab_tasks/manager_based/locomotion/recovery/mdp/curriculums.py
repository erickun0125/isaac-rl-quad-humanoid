# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Recovery-specific curriculum terms for the MDP."""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_recovery(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Curriculum based on the distance the robot walks when recovering.

    This function returns the distance walked by the robot in the x-direction during recovery.
    This can be used to increase the difficulty of the terrain as the robot becomes better
    at recovery tasks.

    Args:
        env: The environment instance.
        asset_cfg: Scene entity configuration for the asset. Defaults to SceneEntityCfg("robot").

    Returns:
        The distance walked by the robot in the x-direction during the current episode.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    
    # compute the distance traveled in the x-direction
    distance = torch.norm(asset.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    
    return distance
