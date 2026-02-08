"""RSL-RL PPO runner configurations for G1 whole-body control."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg
from isaaclab_rl.rsl_rl.symmetry_cfg import RslRlSymmetryCfg
from .symmetry import g1_wholebody_symmetry


@configclass
class G1WholeBodyPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Runner configuration for G1 whole body control with RSL-RL PPO."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 50
    experiment_name = "g1_whole_body_control"
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

    experiment_name = "g1_whole_body_half_rl"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        self.policy.class_name = "ActorCritic"


@configclass
class G1FullRLPPORunnerCfg(G1WholeBodyPPORunnerCfg):
    """Configuration optimized for full RL control (all joints)."""

    experiment_name = "g1_whole_body_full_rl"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    def __post_init__(self):
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        self.policy.class_name = "ActorCritic"
