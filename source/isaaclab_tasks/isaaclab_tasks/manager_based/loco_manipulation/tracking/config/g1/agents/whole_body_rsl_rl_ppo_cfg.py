# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class G1WholeBodyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for G1 whole body control with RSL-RL PPO."""
    
    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    experiment_name = "g1_whole_body_control"
    wandb_project = "isaaclab_g1_whole_body"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class G1LowerBodyOnlyPPORunnerCfg(G1WholeBodyPPORunnerCfg):
    """Configuration optimized for lower body only control."""
    
    experiment_name = "g1_lower_body_only"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],  # Smaller network for fewer DOF
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    
    def __post_init__(self):
        # Add class_name to policy configuration
        super().__post_init__() if hasattr(super(), '__post_init__') else None
        self.policy.class_name = "ActorCritic"


@configclass
class G1FullRLPPORunnerCfg(G1WholeBodyPPORunnerCfg):
    """Configuration optimized for full RL control (all joints)."""
    
    experiment_name = "g1_full_rl"
    max_iterations = 3000  # More iterations for complex full-body control
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[1024, 512, 256],  # Larger network for many DOF
        critic_hidden_dims=[1024, 512, 256],
        activation="elu",
    )
