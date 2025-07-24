# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2RecoveryRoughEnvCfg


@configclass
class UnitreeGo2RecoveryFlatEnvCfg(UnitreeGo2RecoveryRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Flat terrain specific settings only
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None



@configclass
class UnitreeGo2RecoveryFlatEnvCfg_PLAY(UnitreeGo2RecoveryFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
