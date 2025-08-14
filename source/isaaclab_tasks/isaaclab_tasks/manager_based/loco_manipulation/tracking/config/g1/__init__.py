# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for Unitree G1 Humanoid Locomanipulation tasks."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
gym.register(
    id="Isaac-Tracking-LocoManip-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:G1LocoManipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocoManipPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Tracking-LocoManip-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:G1LocoManipEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:G1LocoManipPPORunnerCfg",
    },
)

# Custom enhanced environments with asymmetric observations and custom agent
gym.register(
    id="Isaac-Tracking-LocoManip-G1-Custom-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_loco_manip_env_cfg:G1CustomLocoManipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.custom_rsl_rl_ppo_cfg:G1CustomLocoManipPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Tracking-LocoManip-G1-Custom-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_loco_manip_env_cfg:G1CustomLocoManipEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.custom_rsl_rl_ppo_cfg:G1CustomLocoManipPPORunnerCfg",
    },
)