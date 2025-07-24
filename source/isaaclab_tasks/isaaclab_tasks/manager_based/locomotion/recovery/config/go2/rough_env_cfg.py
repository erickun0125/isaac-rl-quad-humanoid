# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.locomotion.recovery.recovery_env_cfg import LocomotionRecoveryRoughEnvCfg


@configclass
class UnitreeGo2RecoveryRoughEnvCfg(LocomotionRecoveryRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Rough terrain specific adjustments only
        # Enable terrain curriculum for rough environment
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.max_init_terrain_level = 3  # Start with moderate difficulty


@configclass
class UnitreeGo2RecoveryRoughEnvCfg_PLAY(UnitreeGo2RecoveryRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        # disable randomization for play
        self.observations.policy.enable_corruption = False
