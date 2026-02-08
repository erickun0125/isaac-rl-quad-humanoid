"""Unitree G1 Humanoid Loco-Manipulation Environments."""

import gymnasium as gym
from . import agents  # noqa: F401

gym.register(
    id="Isaac-LocoManip-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1LocoManipEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1LocoManipPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-LocoManip-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1LocoManipEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1LocoManipPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-LocoManip-EE-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_ee:G1LocoManipEEEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg_ee:G1LocoManipPPORunnerCfgWithEE",
    },
)

gym.register(
    id="Isaac-LocoManip-EE-G1-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg_ee:G1LocoManipEEEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg_ee:G1LocoManipPPORunnerCfgWithEE",
    },
)
