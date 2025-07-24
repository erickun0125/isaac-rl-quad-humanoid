# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl.symmetry_cfg import RslRlSymmetryCfg
from .symmetry import custom_locomotion_symmetry


@configclass
class UnitreeGo2RecoveryRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32  # Longer episodes for recovery training
    max_iterations = 50000
    save_interval = 500
    experiment_name = "unitree_go2_sequor_recovery_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        noise_std_type="scalar",
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=0.5,
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
class UnitreeGo2SequorRecoveryPPORunnerCfg(UnitreeGo2RecoveryRoughPPORunnerCfg):
    def __post_init__(self):
        self.max_iterations = 50000  # Faster convergence on flat terrain
        self.experiment_name = "unitree_go2_sequor_recovery"
        self.policy.actor_hidden_dims = [256, 256, 128]  # Smaller network for simpler task
        self.policy.critic_hidden_dims = [256, 256, 128]
        self.algorithm.learning_rate = 5.0e-4
        '''
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=True,
            data_augmentation_func=custom_locomotion_symmetry,
            mirror_loss_coeff=0.1,
        )
        '''
