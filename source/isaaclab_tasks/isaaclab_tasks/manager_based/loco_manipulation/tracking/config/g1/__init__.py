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
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_manip_rsl_rl_ppo_cfg:G1LocoManipPPORunnerCfg",
    },
)


gym.register(
    id="Isaac-Tracking-LocoManip-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manip_env_cfg:G1LocoManipEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.loco_manip_rsl_rl_ppo_cfg:G1LocoManipPPORunnerCfg",
    },
)

# Whole body control environments with multi-policy support
# Note: G1WholeBodyEnvCfg and G1WholeBodyEnvCfg_PLAY are base classes, not for direct execution

# Actual executable whole body configurations
gym.register(
    id="Isaac-Tracking-WholeBody-G1-UpperBodyIK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:G1WholeBodyEnvCfg_UpperBodyIK",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.whole_body_rsl_rl_ppo_cfg:G1LowerBodyOnlyPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Tracking-WholeBody-G1-UpperBodyIL-v0", 
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:G1WholeBodyEnvCfg_UpperBodyIL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.whole_body_rsl_rl_ppo_cfg:G1LowerBodyOnlyPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Tracking-WholeBody-G1-FullRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:G1WholeBodyEnvCfg_FullRL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.whole_body_rsl_rl_ppo_cfg:G1FullRLPPORunnerCfg",
    },
)

# Play versions for interactive testing
gym.register(
    id="Isaac-Tracking-WholeBody-G1-UpperBodyIK-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:G1WholeBodyEnvCfg_UpperBodyIK_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.whole_body_rsl_rl_ppo_cfg:G1LowerBodyOnlyPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Tracking-WholeBody-G1-UpperBodyIL-Play-v0", 
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:G1WholeBodyEnvCfg_UpperBodyIL_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.whole_body_rsl_rl_ppo_cfg:G1LowerBodyOnlyPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Tracking-WholeBody-G1-FullRL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv", 
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_env_cfg:G1WholeBodyEnvCfg_FullRL_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.whole_body_rsl_rl_ppo_cfg:G1FullRLPPORunnerCfg",
    },
)