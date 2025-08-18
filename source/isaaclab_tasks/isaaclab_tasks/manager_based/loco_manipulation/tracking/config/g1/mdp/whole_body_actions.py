# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Whole body action classes for G1 with multi-policy support.

This module implements a whole body action system that supports:
- 4 joint groups: Hand, Arm, Waist, Leg
- 3 policy types per group: RL, IL, IK
- Upper Body = Hand + Arm
- Lower Body = Waist + Leg
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, List
from enum import Enum

from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from ..upper_body_IK import UpperBodyIKController, CircularTrajectoryGenerator
from ..upper_body_IL import UpperBodyILController, DummyILModel

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
    
    This action term supports:
    1. 4 joint groups: Hand, Arm, Waist, Leg
    2. 3 policy types per group: RL, IL, IK
    3. RL actions come from step() function
    4. IL and IK are managed internally
    """

    cfg: WholeBodyJointPositionActionCfg
    """Configuration for the action term."""

    def __init__(self, cfg: WholeBodyJointPositionActionCfg, env: ManagerBasedRLEnv) -> None:
        # initialize the action term
        super().__init__(cfg, env)

        # resolve the joints over which the action will be applied
        self._asset: Articulation = env.scene[cfg.asset_name]
        
        # Define joint groups
        self._joint_groups = {
            JointGroup.HAND: cfg.hand_joint_names,
            JointGroup.ARM: cfg.arm_joint_names,
            JointGroup.WAIST: cfg.waist_joint_names,
            JointGroup.LEG: cfg.leg_joint_names,
        }
        
        # Resolve joint indices for each group
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

        # Store policy configuration for each group
        self._group_policies = {
            JointGroup.HAND: cfg.hand_policy,
            JointGroup.ARM: cfg.arm_policy,
            JointGroup.WAIST: cfg.waist_policy,
            JointGroup.LEG: cfg.leg_policy,
        }

        # Create combined indices for all controlled joints
        all_indices = []
        for group in [JointGroup.HAND, JointGroup.ARM, JointGroup.WAIST, JointGroup.LEG]:
            if len(self._joint_group_indices[group]) > 0:
                all_indices.append(self._joint_group_indices[group])
        
        if all_indices:
            # Ensure all indices are tensors
            tensor_indices = []
            for indices in all_indices:
                if isinstance(indices, torch.Tensor):
                    tensor_indices.append(indices)
                else:
                    tensor_indices.append(torch.tensor(indices, dtype=torch.long, device=self.device))
            self._all_controlled_joint_ids = torch.cat(tensor_indices)
        else:
            self._all_controlled_joint_ids = torch.tensor([], dtype=torch.long, device=self.device)

        # Count RL-controlled joints for action dimension
        self._rl_joint_indices = []
        self._rl_joint_count = 0
        
        for group, policy in self._group_policies.items():
            if policy == PolicyType.RL and len(self._joint_group_indices[group]) > 0:
                self._rl_joint_indices.append(self._joint_group_indices[group])
                self._rl_joint_count += len(self._joint_group_indices[group])
        
        if self._rl_joint_indices:
            # Ensure all indices are tensors
            tensor_indices = []
            for indices in self._rl_joint_indices:
                if isinstance(indices, torch.Tensor):
                    tensor_indices.append(indices)
                else:
                    tensor_indices.append(torch.tensor(indices, dtype=torch.long, device=self.device))
            self._rl_joint_indices = torch.cat(tensor_indices)
        else:
            self._rl_joint_indices = torch.tensor([], dtype=torch.long, device=self.device)

        # Prepare scaling and offset for RL joints only
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

            # Set the default joint positions for RL joints
            if cfg.use_default_offset:
                default_joint_pos = self._asset.data.default_joint_pos[:, self._rl_joint_indices].clone()
                self._offset = default_joint_pos[0]  # Take default pose from first env
        else:
            self._scale = torch.tensor([], device=self.device)
            self._offset = torch.tensor([], device=self.device)

        # Initialize controllers for IL and IK policies
        self._ik_controller = None
        self._il_controller = None
        
        # Check if we need IK controller
        if PolicyType.IK in [self._group_policies[JointGroup.HAND], self._group_policies[JointGroup.ARM]]:
            trajectory_generator = CircularTrajectoryGenerator(device=self.device)
            
            # Get URDF path for Pink IK (if available)
            urdf_path = getattr(cfg, 'urdf_path', None)
            mesh_path = getattr(cfg, 'mesh_path', None)
            
            self._ik_controller = UpperBodyIKController(
                robot=self._asset,
                trajectory_generator=trajectory_generator,
                device=self.device,
                urdf_path=urdf_path,
                mesh_path=mesh_path,
            )
            
        # Check if we need IL controller
        if PolicyType.IL in [self._group_policies[JointGroup.HAND], self._group_policies[JointGroup.ARM]]:
            # Create dummy IL models for now
            arm_model = DummyILModel(output_dim=len(self._joint_group_indices[JointGroup.ARM]), device=self.device)
            hand_model = DummyILModel(output_dim=len(self._joint_group_indices[JointGroup.HAND]), device=self.device)
            
            self._il_controller = UpperBodyILController(
                robot=self._asset,
                arm_model=arm_model,
                hand_model=hand_model,
                device=self.device
            )

        # Initialize action buffers
        self._rl_joint_pos_target = torch.zeros(self.num_envs, self._rl_joint_count, device=self.device)
        self._all_joint_pos_target = torch.zeros(self.num_envs, len(self._all_controlled_joint_ids), device=self.device)
        
        # Initialize raw actions buffer
        self._raw_actions = torch.zeros(self.num_envs, self._rl_joint_count, device=self.device)
        
        # Track simulation time for controllers
        self._sim_time = 0.0

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
        # store the raw actions
        self._raw_actions = actions.clone()
        
        # Update simulation time
        dt = self._env.step_dt
        self._sim_time += dt

        # Process RL actions if any
        if self._rl_joint_count > 0:
            self._rl_joint_pos_target = self._scale * self._raw_actions + self._offset
        
        # Combine actions from all policy types for each group
        all_targets = []
        
        for group in [JointGroup.HAND, JointGroup.ARM, JointGroup.WAIST, JointGroup.LEG]:
            if len(self._joint_group_indices[group]) == 0:
                continue  # Skip empty groups
                
            policy_type = self._group_policies[group]
            group_size = len(self._joint_group_indices[group])
            
            if policy_type == PolicyType.RL:
                # Extract RL targets for this group
                group_targets = self._extract_rl_targets_for_group(group)
                
            elif policy_type == PolicyType.IK:
                # Get IK targets for this group
                if group == JointGroup.ARM and self._ik_controller is not None:
                    # Get current arm joint positions for Pink IK
                    current_arm_joints = self._get_current_arm_joint_positions()
                    ik_targets = self._ik_controller.compute_arm_targets(self._sim_time, current_arm_joints)
                    group_targets = ik_targets.unsqueeze(0).expand(self.num_envs, -1)
                elif group == JointGroup.HAND and self._ik_controller is not None:
                    ik_targets = self._ik_controller.compute_hand_targets(self._sim_time)
                    group_targets = ik_targets.unsqueeze(0).expand(self.num_envs, -1)
                else:
                    # Default pose for non-supported IK groups
                    group_targets = torch.zeros(self.num_envs, group_size, device=self.device)
                    
            elif policy_type == PolicyType.IL:
                # Get IL targets for this group
                if self._il_controller is not None:
                    # Get current observations for IL model
                    # This would typically include current state, visual observations, etc.
                    dummy_obs = {"dummy": torch.zeros(self.num_envs, 1, device=self.device)}
                    
                    if group == JointGroup.ARM:
                        il_targets = self._il_controller.compute_arm_targets(dummy_obs)
                        group_targets = il_targets
                    elif group == JointGroup.HAND:
                        il_targets = self._il_controller.compute_hand_targets(dummy_obs)
                        group_targets = il_targets
                    else:
                        # Default pose for non-supported IL groups
                        group_targets = torch.zeros(self.num_envs, group_size, device=self.device)
                else:
                    # Default pose if no IL controller
                    group_targets = torch.zeros(self.num_envs, group_size, device=self.device)
                    
            else:
                # Default case
                group_targets = torch.zeros(self.num_envs, group_size, device=self.device)
                
            all_targets.append(group_targets)
            
        # Combine all group targets
        if all_targets:
            self._all_joint_pos_target = torch.cat(all_targets, dim=-1)
        else:
            self._all_joint_pos_target = torch.zeros(self.num_envs, 0, device=self.device)

    def _extract_rl_targets_for_group(self, group: JointGroup) -> torch.Tensor:
        """Extract RL targets for a specific group from processed RL actions.
        
        Args:
            group: Joint group to extract targets for
            
        Returns:
            Target joint positions for the specified group
        """
        # Find the indices of this group within the RL-controlled joints
        group_indices = self._joint_group_indices[group]
        
        # Find where these indices appear in the RL joint indices
        rl_group_mask = torch.isin(self._rl_joint_indices, group_indices)
        
        if rl_group_mask.any():
            return self._rl_joint_pos_target[:, rl_group_mask]
        else:
            # Return zeros if this group is not RL-controlled
            return torch.zeros(self.num_envs, len(group_indices), device=self.device)
    
    def _get_current_arm_joint_positions(self) -> np.ndarray:
        """Get current arm joint positions for Pink IK solver.
        
        Returns:
            Current arm joint positions as numpy array (for first environment)
        """
        # Get arm joint indices
        arm_joint_indices = self._joint_group_indices[JointGroup.ARM]
        
        if len(arm_joint_indices) > 0:
            # Get current joint positions for arm joints (use first environment)
            current_arm_pos = self._asset.data.joint_pos[0, arm_joint_indices]
            return current_arm_pos.cpu().numpy()
        else:
            # Return empty array if no arm joints
            return np.array([])

    def apply_actions(self) -> None:
        """Apply the processed actions to the articulation."""
        if len(self._all_controlled_joint_ids) > 0:
            self._asset.set_joint_position_target(
                self._all_joint_pos_target, joint_ids=self._all_controlled_joint_ids
            )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Reset the action term for given environment indices.
        
        Args:
            env_ids: Environment indices to reset.
        """
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
            
        # Reset RL joint targets to default positions
        if self._rl_joint_count > 0:
            self._rl_joint_pos_target[env_ids] = self._offset
            
        # Reset all joint targets
        if len(self._all_controlled_joint_ids) > 0:
            # Set to default poses for all groups
            all_defaults = []
            for group in [JointGroup.HAND, JointGroup.ARM, JointGroup.WAIST, JointGroup.LEG]:
                if len(self._joint_group_indices[group]) > 0:
                    # Get default pose for this group
                    group_defaults = self._asset.data.default_joint_pos[:, self._joint_group_indices[group]]
                    all_defaults.append(group_defaults[env_ids])
                    
            if all_defaults:
                self._all_joint_pos_target[env_ids] = torch.cat(all_defaults, dim=-1)

    def get_group_policy(self, group: JointGroup) -> PolicyType:
        """Get the policy type for a specific joint group.
        
        Args:
            group: Joint group to query
            
        Returns:
            Policy type for the group
        """
        return self._group_policies[group]
        
    def get_group_joint_names(self, group: JointGroup) -> List[str]:
        """Get joint names for a specific group.
        
        Args:
            group: Joint group to query
            
        Returns:
            List of joint names for the group
        """
        return self._joint_group_names[group]


@configclass
class WholeBodyJointPositionActionCfg(ActionTermCfg):
    """Configuration for whole body joint position action with multi-policy support."""
    
    class_type: type = WholeBodyJointPositionAction
    """Class type for the action term."""

    asset_name: str = "robot"
    """Name of the asset in the environment for which the action term is being applied."""
    
    # Joint group names
    hand_joint_names: List[str] = None
    """List of hand joint names or regex expressions."""
    
    arm_joint_names: List[str] = None  
    """List of arm joint names or regex expressions."""
    
    waist_joint_names: List[str] = None
    """List of waist joint names or regex expressions."""
    
    leg_joint_names: List[str] = None
    """List of leg joint names or regex expressions."""

    # Policy types for each group
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

    # Pink IK configuration (optional)
    urdf_path: str | None = None
    """Path to URDF file for Pink IK controller. If provided, Pink IK will be used for arm control."""
    
    mesh_path: str | None = None
    """Path to mesh files for Pink IK controller."""
