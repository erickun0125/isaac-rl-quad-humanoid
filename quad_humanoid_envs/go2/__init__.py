"""Unitree GO2 quadruped robot environments: velocity tracking and fall recovery."""

import gymnasium as gym

from . import velocity  # noqa: F401
from . import recovery  # noqa: F401

# =============================================================================
# Velocity: Flat terrain
# =============================================================================

gym.register(
    id="Isaac-Velocity-Sequor-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "quad_humanoid_envs.go2.velocity.env_cfg:UnitreeGo2SequorEnvCfg",
        "rsl_rl_cfg_entry_point": "quad_humanoid_envs.go2.velocity.agents.ppo_cfg:UnitreeGo2SequorPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Sequor-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "quad_humanoid_envs.go2.velocity.env_cfg:UnitreeGo2SequorEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "quad_humanoid_envs.go2.velocity.agents.ppo_cfg:UnitreeGo2SequorPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Sequor-RNN-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "quad_humanoid_envs.go2.velocity.env_cfg:UnitreeGo2SequorEnvCfg",
        "rsl_rl_cfg_entry_point": "quad_humanoid_envs.go2.velocity.agents.ppo_cfg:UnitreeGo2SequorRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Sequor-RNN-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "quad_humanoid_envs.go2.velocity.env_cfg:UnitreeGo2SequorEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "quad_humanoid_envs.go2.velocity.agents.ppo_cfg:UnitreeGo2SequorRNNPPORunnerCfg",
    },
)

# =============================================================================
# Velocity: Rough terrain
# =============================================================================

gym.register(
    id="Isaac-Velocity-Sequor-Rough-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "quad_humanoid_envs.go2.velocity.env_cfg:UnitreeGo2SequorRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "quad_humanoid_envs.go2.velocity.agents.ppo_cfg:UnitreeGo2SequorPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Sequor-Rough-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "quad_humanoid_envs.go2.velocity.env_cfg:UnitreeGo2SequorRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "quad_humanoid_envs.go2.velocity.agents.ppo_cfg:UnitreeGo2SequorPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Sequor-Rough-RNN-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "quad_humanoid_envs.go2.velocity.env_cfg:UnitreeGo2SequorRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "quad_humanoid_envs.go2.velocity.agents.ppo_cfg:UnitreeGo2SequorRNNPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Velocity-Sequor-Rough-RNN-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "quad_humanoid_envs.go2.velocity.env_cfg:UnitreeGo2SequorRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "quad_humanoid_envs.go2.velocity.agents.ppo_cfg:UnitreeGo2SequorRNNPPORunnerCfg",
    },
)

# =============================================================================
# Recovery: Flat terrain
# =============================================================================

gym.register(
    id="Isaac-Recovery-Sequor-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "quad_humanoid_envs.go2.recovery.env_cfg:UnitreeGo2RecoveryFlatEnvCfg",
        "rsl_rl_cfg_entry_point": "quad_humanoid_envs.go2.recovery.agents.ppo_cfg:UnitreeGo2SequorRecoveryPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Recovery-Sequor-Flat-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "quad_humanoid_envs.go2.recovery.env_cfg:UnitreeGo2RecoveryFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "quad_humanoid_envs.go2.recovery.agents.ppo_cfg:UnitreeGo2SequorRecoveryPPORunnerCfg",
    },
)

# =============================================================================
# Recovery: Rough terrain
# =============================================================================

gym.register(
    id="Isaac-Recovery-Sequor-Rough-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "quad_humanoid_envs.go2.recovery.env_cfg:UnitreeGo2RecoveryRoughEnvCfg",
        "rsl_rl_cfg_entry_point": "quad_humanoid_envs.go2.recovery.agents.ppo_cfg:UnitreeGo2RecoveryRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Recovery-Sequor-Rough-Unitree-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "quad_humanoid_envs.go2.recovery.env_cfg:UnitreeGo2RecoveryRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "quad_humanoid_envs.go2.recovery.agents.ppo_cfg:UnitreeGo2RecoveryRoughPPORunnerCfg",
    },
)
