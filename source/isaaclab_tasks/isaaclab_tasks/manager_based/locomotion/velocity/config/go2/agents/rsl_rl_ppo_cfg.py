# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "unitree_go2_rough"
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
class UnitreeGo2FlatPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        self.max_iterations = 300
        self.experiment_name = "unitree_go2_flat"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [128, 128, 128]

@configclass
class UnitreeGo2SequorPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        self.save_interval = 500
        self.max_iterations = 15000
        self.experiment_name = "unitree_go2_sequor"
        self.policy.actor_hidden_dims = [128, 128, 128]
        self.policy.critic_hidden_dims = [256, 128, 128]
        self.algorithm.learning_rate = 5.0e-4


@configclass
class UnitreeGo2SequorRNNPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        self.save_interval = 500
        self.max_iterations = 15000
        self.experiment_name = "unitree_go2_sequor_rnn"
        
        # RNN 버전의 policy 설정
        self.policy = RslRlPpoActorCriticRecurrentCfg(
            noise_std_type="scalar",
            init_noise_std=1.0,
            actor_hidden_dims=[128, 128, 128],
            critic_hidden_dims=[256, 128, 128],
            activation="elu",
            rnn_type="gru",
            rnn_hidden_dim=256,
            rnn_num_layers=1,
        )
        
        # 학습률을 조정
        self.algorithm.learning_rate = 5.0e-4
