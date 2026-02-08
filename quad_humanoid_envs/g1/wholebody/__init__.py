"""G1 humanoid whole-body control environments."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-WholeBody-G1-UpperBodyIK-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1WholeBodyEnvCfg_UpperBodyIK",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1LowerBodyOnlyPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WholeBody-G1-UpperBodyIL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1WholeBodyEnvCfg_UpperBodyIL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1LowerBodyOnlyPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WholeBody-G1-FullRL-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1WholeBodyEnvCfg_FullRL",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1FullRLPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WholeBody-G1-UpperBodyIK-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1WholeBodyEnvCfg_UpperBodyIK_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1LowerBodyOnlyPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WholeBody-G1-UpperBodyIL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1WholeBodyEnvCfg_UpperBodyIL_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1LowerBodyOnlyPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-WholeBody-G1-FullRL-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:G1WholeBodyEnvCfg_FullRL_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_cfg:G1FullRLPPORunnerCfg",
    },
)
