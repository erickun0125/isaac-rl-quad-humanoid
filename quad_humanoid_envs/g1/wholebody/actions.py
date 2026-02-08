"""Whole body action classes for G1 with multi-policy support.

Implements a whole body action system that supports:
- 4 joint groups: Hand, Arm, Waist, Leg
- 3 policy types per group: RL, IL, IK
- Upper Body = Hand + Arm
- Lower Body = Waist + Leg
"""

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING, List
from enum import Enum

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from .controllers import (
    CircularTrajectoryGenerator,
    UpperBodyIKController,
    UpperBodyILController,
    DummyILModel,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class PolicyType(Enum):
    """Enumeration for policy types."""
    RL = "rl"
    IL = "il"
    IK = "ik"


class JointGroup(Enum):
    """Enumeration for joint groups."""
    HAND = "hand"
    ARM = "arm"
    WAIST = "waist"
    LEG = "leg"


class WholeBodyJointPositionAction(ActionTerm):
    """Whole body joint position action with multi-policy support.

    Supports:
    1. 4 joint groups: Hand, Arm, Waist, Leg
    2. 3 policy types per group: RL, IL, IK
    3. RL actions come from step() function
    4. IL and IK are managed internally
    """

    cfg: WholeBodyJointPositionActionCfg

    def __init__(self, cfg: WholeBodyJointPositionActionCfg, env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]

        self._joint_groups = {
            JointGroup.HAND: cfg.hand_joint_names,
            JointGroup.ARM: cfg.arm_joint_names,
            JointGroup.WAIST: cfg.waist_joint_names,
            JointGroup.LEG: cfg.leg_joint_names,
        }

        self._joint_group_indices = {}
        self._joint_group_names = {}

        for group, joint_names in self._joint_groups.items():
            if joint_names:
                joint_ids, joint_names_resolved = self._asset.find_joints(joint_names)
                self._joint_group_indices[group] = joint_ids
                self._joint_group_names[group] = joint_names_resolved
                print(f"[INFO] Resolved {group.value} joints ({len(joint_ids)}): {joint_names_resolved}")
            else:
                self._joint_group_indices[group] = torch.tensor([], dtype=torch.long, device=self.device)
                self._joint_group_names[group] = []

        self._group_policies = {
            JointGroup.HAND: cfg.hand_policy,
            JointGroup.ARM: cfg.arm_policy,
            JointGroup.WAIST: cfg.waist_policy,
            JointGroup.LEG: cfg.leg_policy,
        }

        all_indices = []
        for group in [JointGroup.HAND, JointGroup.ARM, JointGroup.WAIST, JointGroup.LEG]:
            if len(self._joint_group_indices[group]) > 0:
                all_indices.append(self._joint_group_indices[group])

        if all_indices:
            tensor_indices = []
            for indices in all_indices:
                if isinstance(indices, torch.Tensor):
                    tensor_indices.append(indices)
                else:
                    tensor_indices.append(torch.tensor(indices, dtype=torch.long, device=self.device))
            self._all_controlled_joint_ids = torch.cat(tensor_indices)
        else:
            self._all_controlled_joint_ids = torch.tensor([], dtype=torch.long, device=self.device)

        self._rl_joint_indices = []
        self._rl_joint_count = 0

        for group, policy in self._group_policies.items():
            if policy == PolicyType.RL and len(self._joint_group_indices[group]) > 0:
                self._rl_joint_indices.append(self._joint_group_indices[group])
                self._rl_joint_count += len(self._joint_group_indices[group])

        if self._rl_joint_indices:
            tensor_indices = []
            for indices in self._rl_joint_indices:
                if isinstance(indices, torch.Tensor):
                    tensor_indices.append(indices)
                else:
                    tensor_indices.append(torch.tensor(indices, dtype=torch.long, device=self.device))
            self._rl_joint_indices = torch.cat(tensor_indices)
        else:
            self._rl_joint_indices = torch.tensor([], dtype=torch.long, device=self.device)

        if self._rl_joint_count > 0:
            self._scale = cfg.scale
            if isinstance(cfg.scale, (float, int)):
                self._scale = torch.full((self._rl_joint_count,), cfg.scale, device=self.device)
            else:
                self._scale = torch.tensor(cfg.scale[:self._rl_joint_count], device=self.device)

            self._offset = cfg.offset
            if isinstance(cfg.offset, (float, int)):
                self._offset = torch.full((self._rl_joint_count,), cfg.offset, device=self.device)
            else:
                self._offset = torch.tensor(cfg.offset[:self._rl_joint_count], device=self.device)

            if cfg.use_default_offset:
                default_joint_pos = self._asset.data.default_joint_pos[:, self._rl_joint_indices].clone()
                self._offset = default_joint_pos[0]
        else:
            self._scale = torch.tensor([], device=self.device)
            self._offset = torch.tensor([], device=self.device)

        self._ik_controller = None
        self._il_controller = None

        if PolicyType.IK in [self._group_policies[JointGroup.HAND], self._group_policies[JointGroup.ARM]]:
            trajectory_generator = self._create_trajectory_generator(cfg)

            urdf_path = getattr(cfg, "urdf_path", None)
            mesh_path = getattr(cfg, "mesh_path", None)

            self._ik_controller = UpperBodyIKController(
                robot=self._asset,
                trajectory_generator=trajectory_generator,
                device=self.device,
                urdf_path=urdf_path,
                mesh_path=mesh_path,
            )

        if PolicyType.IL in [self._group_policies[JointGroup.HAND], self._group_policies[JointGroup.ARM]]:
            self._il_controller = self._create_il_controller(cfg)

        self._rl_joint_pos_target = torch.zeros(self.num_envs, self._rl_joint_count, device=self.device)
        self._all_joint_pos_target = torch.zeros(self.num_envs, len(self._all_controlled_joint_ids), device=self.device)
        self._raw_actions = torch.zeros(self.num_envs, self._rl_joint_count, device=self.device)
        self._sim_time = 0.0

        print(f"[INFO] WholeBodyJointPositionAction initialized with {self._rl_joint_count} RL-controlled joints")
        print(f"[INFO] Action dimension: {self._rl_joint_count}")

    def _create_trajectory_generator(self, cfg: WholeBodyJointPositionActionCfg):
        """Create trajectory generator based on configuration."""
        trajectory_type = getattr(cfg, "trajectory_generator_type", "circular")
        trajectory_params = getattr(cfg, "trajectory_generator_params", None) or {}

        if trajectory_type == "circular":
            return CircularTrajectoryGenerator(device=self.device, **trajectory_params)
        else:
            print(f"[WARNING] Trajectory generator type '{trajectory_type}' not implemented, using circular")
            return CircularTrajectoryGenerator(device=self.device, **trajectory_params)

    def _create_il_controller(self, cfg: WholeBodyJointPositionActionCfg) -> UpperBodyILController:
        """Create IL controller based on configuration."""
        policy_type = getattr(cfg, "upper_body_policy_type", "separate")
        model_path = getattr(cfg, "upper_body_policy_model_path", None)

        if policy_type == "unified":
            upper_body_dim = len(self._joint_group_indices[JointGroup.ARM]) + len(self._joint_group_indices[JointGroup.HAND])
            upper_body_model = DummyILModel(output_dim=upper_body_dim, device=self.device)

            if model_path:
                try:
                    upper_body_model.load_model(model_path)
                    print(f"[INFO] Loaded unified upper body IL model from {model_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to load unified IL model: {e}")

            return UpperBodyILController(
                robot=self._asset,
                upper_body_model=upper_body_model,
                policy_type="unified",
                device=self.device,
            )
        else:
            arm_model = DummyILModel(output_dim=len(self._joint_group_indices[JointGroup.ARM]), device=self.device)
            hand_model = DummyILModel(output_dim=len(self._joint_group_indices[JointGroup.HAND]), device=self.device)

            if model_path:
                try:
                    arm_model.load_model(f"{model_path}/arm_model.pt")
                    hand_model.load_model(f"{model_path}/hand_model.pt")
                    print(f"[INFO] Loaded separate arm/hand IL models from {model_path}")
                except Exception as e:
                    print(f"[WARNING] Failed to load separate IL models: {e}")

            return UpperBodyILController(
                robot=self._asset,
                arm_model=arm_model,
                hand_model=hand_model,
                policy_type="separate",
                device=self.device,
            )

    @property
    def action_dim(self) -> int:
        """Dimension of the action space (RL-controlled joints only)."""
        return self._rl_joint_count

    @property
    def raw_actions(self) -> torch.Tensor:
        """The raw actions received from RL policy."""
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """The processed actions for all controlled joints."""
        return self._all_joint_pos_target

    def process_actions(self, actions: torch.Tensor) -> None:
        """Process the raw actions from RL policy and combine with IL/IK controllers.

        Args:
            actions: Raw actions from RL policy for RL-controlled joints only.
        """
        self._raw_actions = actions.clone()

        dt = self._env.step_dt
        self._sim_time += dt

        if self._rl_joint_count > 0:
            self._rl_joint_pos_target = self._scale * self._raw_actions + self._offset

        all_targets = []

        for group in [JointGroup.HAND, JointGroup.ARM, JointGroup.WAIST, JointGroup.LEG]:
            if len(self._joint_group_indices[group]) == 0:
                continue

            policy_type = self._group_policies[group]
            group_size = len(self._joint_group_indices[group])

            if policy_type == PolicyType.RL:
                group_targets = self._extract_rl_targets_for_group(group)

            elif policy_type == PolicyType.IK:
                if group == JointGroup.ARM and self._ik_controller is not None:
                    current_arm_joints = self._get_current_arm_joint_positions()
                    ik_targets = self._ik_controller.compute_arm_targets(self._sim_time, current_arm_joints)
                    group_targets = ik_targets.unsqueeze(0).expand(self.num_envs, -1)
                elif group == JointGroup.HAND and self._ik_controller is not None:
                    ik_targets = self._ik_controller.compute_hand_targets(self._sim_time)
                    group_targets = ik_targets.unsqueeze(0).expand(self.num_envs, -1)
                else:
                    group_targets = torch.zeros(self.num_envs, group_size, device=self.device)

            elif policy_type == PolicyType.IL:
                if self._il_controller is not None:
                    dummy_obs = {"dummy": torch.zeros(self.num_envs, 1, device=self.device)}

                    if group == JointGroup.ARM:
                        il_targets = self._il_controller.compute_arm_targets(dummy_obs)
                        group_targets = il_targets
                    elif group == JointGroup.HAND:
                        il_targets = self._il_controller.compute_hand_targets(dummy_obs)
                        group_targets = il_targets
                    else:
                        group_targets = torch.zeros(self.num_envs, group_size, device=self.device)
                else:
                    group_targets = torch.zeros(self.num_envs, group_size, device=self.device)

            else:
                group_targets = torch.zeros(self.num_envs, group_size, device=self.device)

            all_targets.append(group_targets)

        if all_targets:
            self._all_joint_pos_target = torch.cat(all_targets, dim=-1)
        else:
            self._all_joint_pos_target = torch.zeros(self.num_envs, 0, device=self.device)

    def _extract_rl_targets_for_group(self, group: JointGroup) -> torch.Tensor:
        """Extract RL targets for a specific group from processed RL actions."""
        group_indices = self._joint_group_indices[group]

        if isinstance(group_indices, list):
            group_indices_tensor = torch.tensor(group_indices, device=self.device, dtype=torch.long)
        else:
            group_indices_tensor = group_indices

        rl_group_mask = torch.isin(self._rl_joint_indices, group_indices_tensor)

        if rl_group_mask.any():
            return self._rl_joint_pos_target[:, rl_group_mask]
        else:
            if isinstance(group_indices, list):
                group_size = len(group_indices)
            else:
                group_size = group_indices.numel()
            return torch.zeros(self.num_envs, group_size, device=self.device)

    def _get_current_arm_joint_positions(self) -> np.ndarray:
        """Get current arm joint positions for Pink IK solver."""
        arm_joint_indices = self._joint_group_indices[JointGroup.ARM]

        if len(arm_joint_indices) > 0:
            current_arm_pos = self._asset.data.joint_pos[0, arm_joint_indices]
            return current_arm_pos.cpu().numpy()
        else:
            return np.array([])

    def apply_actions(self) -> None:
        """Apply the processed actions to the articulation."""
        if len(self._all_controlled_joint_ids) > 0:
            self._asset.set_joint_position_target(
                self._all_joint_pos_target, joint_ids=self._all_controlled_joint_ids
            )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the action term for given environment indices."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        if self._rl_joint_count > 0:
            self._rl_joint_pos_target[env_ids] = self._offset

        if len(self._all_controlled_joint_ids) > 0:
            all_defaults = []
            for group in [JointGroup.HAND, JointGroup.ARM, JointGroup.WAIST, JointGroup.LEG]:
                if len(self._joint_group_indices[group]) > 0:
                    group_defaults = self._asset.data.default_joint_pos[:, self._joint_group_indices[group]]
                    all_defaults.append(group_defaults[env_ids])

            if all_defaults:
                self._all_joint_pos_target[env_ids] = torch.cat(all_defaults, dim=-1)

    def get_group_policy(self, group: JointGroup) -> PolicyType:
        """Get the policy type for a specific joint group."""
        return self._group_policies[group]

    def get_group_joint_names(self, group: JointGroup) -> List[str]:
        """Get joint names for a specific group."""
        return self._joint_group_names[group]


@configclass
class WholeBodyJointPositionActionCfg(ActionTermCfg):
    """Configuration for whole body joint position action with multi-policy support."""

    class_type: type = WholeBodyJointPositionAction

    asset_name: str = "robot"
    """Name of the asset in the environment."""

    hand_joint_names: List[str] = None
    """List of hand joint names or regex expressions."""

    arm_joint_names: List[str] = None
    """List of arm joint names or regex expressions."""

    waist_joint_names: List[str] = None
    """List of waist joint names or regex expressions."""

    leg_joint_names: List[str] = None
    """List of leg joint names or regex expressions."""

    hand_policy: PolicyType = PolicyType.IK
    """Policy type for hand control."""

    arm_policy: PolicyType = PolicyType.IK
    """Policy type for arm control."""

    waist_policy: PolicyType = PolicyType.RL
    """Policy type for waist control."""

    leg_policy: PolicyType = PolicyType.RL
    """Policy type for leg control."""

    scale: float | torch.Tensor = 1.0
    """Scale factor for RL actions. Defaults to 1.0."""

    offset: float | torch.Tensor = 0.0
    """Offset factor for RL actions. Defaults to 0.0."""

    use_default_offset: bool = False
    """Whether to use default joint positions as offset for RL joints. Defaults to False."""

    urdf_path: str | None = None
    """Path to URDF file for Pink IK controller."""

    mesh_path: str | None = None
    """Path to mesh files for Pink IK controller."""

    trajectory_generator_type: str = "circular"
    """Type of trajectory generator for IK policy. Options: 'circular', 'linear', 'custom'."""

    trajectory_generator_params: dict | None = None
    """Parameters for the trajectory generator."""

    upper_body_policy_type: str = "separate"
    """Type of upper body IL policy. Options: 'separate', 'unified'."""

    upper_body_policy_model_path: str | None = None
    """Path to the upper body IL policy model file."""
