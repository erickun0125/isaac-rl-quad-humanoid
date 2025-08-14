# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class G1CustomLocoManipPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Custom RSL-RL PPO configuration for G1 locomanipulation with asymmetric observations."""
    
    num_steps_per_env = 24
    max_iterations = 4000  # Increased for more thorough training
    save_interval = 100    # More frequent saves for custom training
    experiment_name = "g1_custom_loco_manip"
    empirical_normalization = False
    
    # Enhanced policy configuration for asymmetric observations
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        # Larger networks to handle increased observation space
        actor_hidden_dims=[512, 512, 256],  # Increased capacity for actor
        critic_hidden_dims=[512, 512, 256], # Increased capacity for critic
        activation="elu",
    )
    
    # Enhanced algorithm configuration for better convergence
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,  # Slightly reduced for more focused exploration
        num_learning_epochs=6,  # Increased for better learning
        num_mini_batches=4,
        learning_rate=8e-4,   # Slightly reduced for stability
        schedule="adaptive",
        gamma=0.998,          # Slightly increased for longer-term rewards
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )