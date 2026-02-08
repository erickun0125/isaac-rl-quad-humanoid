"""GO2 recovery environment configurations (base, flat, and rough terrain)."""

import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.recovery.mdp as mdp

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip


# =============================================================================
# Scene definition
# =============================================================================


@configclass
class RecoverySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot for recovery training."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=3,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )

    robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# =============================================================================
# MDP settings
# =============================================================================


@configclass
class RecoveryActionsCfg:
    """Action specifications for recovery MDP."""

    joint_pos = mdp.RelativeJointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.25)


@configclass
class RecoveryObservationsCfg:
    """Observation specifications for recovery MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1), history_length=2)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            history_length=2,
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=2)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=2)
        actions = ObsTerm(func=mdp.last_action, history_length=2)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.1, n_max=0.1), history_length=2)
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.02, n_max=0.02),
            history_length=2,
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01), history_length=2)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5), history_length=2)
        actions = ObsTerm(func=mdp.last_action, history_length=2)
        progress_ratio = ObsTerm(func=mdp.episode_progress)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class RecoveryEventCfg:
    """Configuration for recovery events."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "com_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03), "z": (-0.005, 0.005)},
        },
    )

    reset_with_physics_simulation = EventTerm(
        func=mdp.reset_with_physics_simulation,
        mode="reset",
        params={
            "pose_range": {"x": (-3.0, 3.0), "y": (-3.0, 3.0), "yaw": (-3.14, 3.14)},
            "roll_range": (-1.5, 1.5),
            "pitch_range": (-1.5, 1.5),
            "height_range": (0.35, 0.55),
            "simulation_time": 0.5,
        },
    )


@configclass
class RecoveryRewardsCfg:
    """Reward terms for recovery MDP."""

    flat_orientation = RewTerm(
        func=mdp.flat_orientation,
        weight=1.0,
    )
    joint_pose = RewTerm(
        func=mdp.joint_pose,
        weight=1.0,
    )
    foot_contact = RewTerm(
        func=mdp.foot_contact,
        weight=0.1,
    )
    final_configuration = RewTerm(
        func=mdp.final_configuration,
        weight=15.0,
    )

    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_threshold_penalty, weight=-0.2, params={"threshold": 0.3})
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_threshold_penalty, weight=-0.1, params={"threshold": 1.0})
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
        },
    )

    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base")},
    )

    head_contact = RewTerm(
        func=mdp.undesired_contacts,
        weight=-20.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Head_upper", "Head_lower"]),
            "threshold": 0.5,
        },
    )


@configclass
class RecoveryTerminationsCfg:
    """Termination terms for recovery MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class RecoveryCurriculumCfg:
    """Curriculum terms for recovery MDP."""

    pass


# =============================================================================
# Base recovery environment
# =============================================================================


@configclass
class LocomotionRecoveryRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for GO2 locomotion recovery environment."""

    scene: RecoverySceneCfg = RecoverySceneCfg(num_envs=512, env_spacing=2.5)
    observations: RecoveryObservationsCfg = RecoveryObservationsCfg()
    actions: RecoveryActionsCfg = RecoveryActionsCfg()
    rewards: RecoveryRewardsCfg = RecoveryRewardsCfg()
    terminations: RecoveryTerminationsCfg = RecoveryTerminationsCfg()
    events: RecoveryEventCfg = RecoveryEventCfg()
    curriculum: RecoveryCurriculumCfg = RecoveryCurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 5.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15

        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
            self.scene.terrain.max_init_terrain_level = 2


# =============================================================================
# Rough terrain recovery
# =============================================================================


@configclass
class UnitreeGo2RecoveryRoughEnvCfg(LocomotionRecoveryRoughEnvCfg):
    """GO2 recovery environment on rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = True
            self.scene.terrain.max_init_terrain_level = 3


@configclass
class UnitreeGo2RecoveryRoughEnvCfg_PLAY(UnitreeGo2RecoveryRoughEnvCfg):
    """Play mode configuration for rough terrain recovery."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False


# =============================================================================
# Flat terrain recovery
# =============================================================================


@configclass
class UnitreeGo2RecoveryFlatEnvCfg(UnitreeGo2RecoveryRoughEnvCfg):
    """GO2 recovery environment on flat terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None


@configclass
class UnitreeGo2RecoveryFlatEnvCfg_PLAY(UnitreeGo2RecoveryFlatEnvCfg):
    """Play mode configuration for flat terrain recovery."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
