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

from ..upper_body_controller import (
    CircularTrajectoryGenerator,
    FixedPointTrajectoryGenerator,
    UpperBodyIKController,
    IKDataCollector,
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

        # Initialize IK data collector for modular data access
        self._ik_data_collector = IKDataCollector(
            asset=self._asset,
            joint_group_indices=self._joint_group_indices,
            joint_group_names=self._joint_group_names,
            device=self.device
        )
        
        # Initialize controllers for IL and IK policies
        self._ik_controller = None
        self._il_controller = None
        
        # Check if we need IK controller
        if PolicyType.IK in [self._group_policies[JointGroup.HAND], self._group_policies[JointGroup.ARM]]:
            # Create trajectory generator based on configuration
            trajectory_generator = self._create_trajectory_generator(cfg)
            
            # Create IK controller (Differential IK only)
            print("Initializing Differential IK controller for upper body")
            self._ik_controller = UpperBodyIKController(
                robot=self._asset,
                trajectory_generator=trajectory_generator,
                device=self.device,
                num_envs=self.num_envs,
            )
            
        # Check if we need IL controller
        if PolicyType.IL in [self._group_policies[JointGroup.HAND], self._group_policies[JointGroup.ARM]]:
            # Create IL models and controller based on configuration
            self._il_controller = self._create_il_controller(cfg)
        else:
            self._il_controller = None

        # Initialize action buffers
        self._rl_joint_pos_target = torch.zeros(self.num_envs, self._rl_joint_count, device=self.device)
        self._all_joint_pos_target = torch.zeros(self.num_envs, len(self._all_controlled_joint_ids), device=self.device)
        
        # Initialize raw actions buffer
        self._raw_actions = torch.zeros(self.num_envs, self._rl_joint_count, device=self.device)
        
        # Track simulation time for controllers
        self._sim_time = 0.0

        # Print initialization summary
        self._print_initialization_summary()
    
    def _create_trajectory_generator(self, cfg: WholeBodyJointPositionActionCfg) -> "TrajectoryGenerator":
        """Create trajectory generator based on configuration.
        
        Args:
            cfg: Action configuration
            
        Returns:
            Configured trajectory generator
        """
        trajectory_type = getattr(cfg, 'trajectory_generator_type', 'circular')
        trajectory_params = getattr(cfg, 'trajectory_generator_params', None) or {}
        
        if trajectory_type == "circular":
            return CircularTrajectoryGenerator(device=self.device, **trajectory_params)
        elif trajectory_type == "fixed_point":
            return FixedPointTrajectoryGenerator(device=self.device, **trajectory_params)
        else:
            # Default to circular for now
            print(f"[WARNING] Trajectory generator type '{trajectory_type}' not implemented, using circular")
            return CircularTrajectoryGenerator(device=self.device, **trajectory_params)
    
    def _print_initialization_summary(self):
        """Print initialization summary for debugging."""
        print("\n" + "="*60)
        print("WholeBodyJointPositionAction Initialization Summary")
        print("="*60)
        
        # Joint group configuration
        for group, policy in self._group_policies.items():
            joint_count = len(self._joint_group_indices[group])
            print(f"  {group.value.upper()}: {policy.value.upper()} policy ({joint_count} joints)")
        
        # Controller information
        print("\nController Status:")
        print(f"  IK Controller: {'✓' if self._ik_controller else '✗'}")
        print(f"  IL Controller: {'✓' if self._il_controller else '✗'}")
        print(f"  IK Data Collector: {'✓' if self._ik_data_collector else '✗'}")
        
        # Action space information
        print("\nAction Space:")
        print(f"  RL-controlled joints: {self._rl_joint_count}")
        print(f"  Total controlled joints: {len(self._all_controlled_joint_ids)}")
        print(f"  Environments: {self.num_envs}")
        
        print("="*60 + "\n")
    
    def _create_il_controller(self, cfg: WholeBodyJointPositionActionCfg) -> UpperBodyILController:
        """Create IL controller based on configuration.
        
        Args:
            cfg: Action configuration
            
        Returns:
            Configured IL controller
        """
        policy_type = getattr(cfg, 'upper_body_policy_type', 'separate')
        model_path = getattr(cfg, 'upper_body_policy_model_path', None)
        
        if policy_type == "unified":
            # Unified mode: single model for both arm and hand
            upper_body_dim = len(self._joint_group_indices[JointGroup.ARM]) + len(self._joint_group_indices[JointGroup.HAND])
            upper_body_model = DummyILModel(output_dim=upper_body_dim, device=self.device)
            
            if model_path:
                try:
                    upper_body_model.load_model(model_path)
                    print(f"[INFO] Loaded unified upper body IL model from {model_path}")
                except Exception as e:  # pylint: disable=broad-except
                    print(f"[WARNING] Failed to load unified IL model: {e}")
            
            return UpperBodyILController(
                robot=self._asset,
                upper_body_model=upper_body_model,
                policy_type="unified",
                device=self.device
            )
        else:
            # Separate mode: individual models for arm and hand
            arm_model = DummyILModel(output_dim=len(self._joint_group_indices[JointGroup.ARM]), device=self.device)
            hand_model = DummyILModel(output_dim=len(self._joint_group_indices[JointGroup.HAND]), device=self.device)
            
            if model_path:
                try:
                    # In separate mode, assume model_path contains both models
                    arm_model.load_model(f"{model_path}/arm_model.pt")
                    hand_model.load_model(f"{model_path}/hand_model.pt")
                    print(f"[INFO] Loaded separate arm/hand IL models from {model_path}")
                except Exception as e:  # pylint: disable=broad-except
                    print(f"[WARNING] Failed to load separate IL models: {e}")
            
            return UpperBodyILController(
                robot=self._asset,
                arm_model=arm_model,
                hand_model=hand_model,
                policy_type="separate",
                device=self.device
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
                    group_targets = self._compute_ik_arm_targets(group_size)
                elif group == JointGroup.HAND and self._ik_controller is not None:
                    group_targets = self._compute_ik_hand_targets(group_size)
                else:
                    # Default pose for non-supported IK groups
                    group_targets = self._get_default_pose(group_size)
                    
            elif policy_type == PolicyType.IL:
                # Get IL targets for this group
                if group == JointGroup.ARM and self._il_controller is not None:
                    group_targets = self._compute_il_arm_targets(group_size)
                elif group == JointGroup.HAND and self._il_controller is not None:
                    group_targets = self._compute_il_hand_targets(group_size)
                else:
                    # Default pose for non-supported IL groups or no IL controller
                    group_targets = self._get_default_pose(group_size)
                    
            else:
                # Default case
                group_targets = torch.zeros(self.num_envs, group_size, device=self.device)
                
            all_targets.append(group_targets)
            
        # Combine all group targets
        if all_targets:
            self._all_joint_pos_target = torch.cat(all_targets, dim=-1)
        else:
            self._all_joint_pos_target = torch.zeros(self.num_envs, 0, device=self.device)

    def _compute_ik_arm_targets(self, group_size: int) -> torch.Tensor:
        """Compute IK targets for arm joints.
        
        Args:
            group_size: Number of joints in the group
            
        Returns:
            Target joint positions for arm joints (num_envs, group_size)
        """
        try:
            # Collect all IK data using the modular data collector
            ik_data = self._ik_data_collector.collect_ik_data()
            
            # Compute IK targets using collected data
            ik_targets = self._ik_controller.compute_arm_targets(
                current_time=self._sim_time,
                left_current_joints=ik_data['left_current_joints'],
                right_current_joints=ik_data['right_current_joints'],
                left_ee_pos=ik_data['left_ee_pos'],
                left_ee_quat=ik_data['left_ee_quat'],
                right_ee_pos=ik_data['right_ee_pos'],
                right_ee_quat=ik_data['right_ee_quat'],
                left_jacobian=ik_data['left_jacobian'],
                right_jacobian=ik_data['right_jacobian']
            )
            
            return ik_targets
            
        except (RuntimeError, ValueError, AttributeError) as e:
            print(f"[WARNING] ARM IK computation failed: {e}. Using default pose.")
            return self._get_default_pose(group_size)
    
    def _compute_ik_hand_targets(self, group_size: int) -> torch.Tensor:
        """Compute IK targets for hand joints.
        
        Args:
            group_size: Number of joints in the group
            
        Returns:
            Target joint positions for hand joints (num_envs, group_size)
        """
        try:
            hand_targets = self._ik_controller.compute_hand_targets(self._sim_time)
            return hand_targets.unsqueeze(0).expand(self.num_envs, -1)
        except (RuntimeError, ValueError, AttributeError) as e:
            print(f"[WARNING] HAND IK computation failed: {e}. Using default pose.")
            return self._get_default_pose(group_size)
    
    def _compute_il_arm_targets(self, group_size: int) -> torch.Tensor:
        """Compute IL targets for arm joints.
        
        Args:
            group_size: Number of joints in the group
            
        Returns:
            Target joint positions for arm joints (num_envs, group_size)
        """
        try:
            # Get current observations for IL model
            # This would typically include current state, visual observations, etc.
            dummy_obs = {"dummy": torch.zeros(self.num_envs, 1, device=self.device)}
            
            il_targets = self._il_controller.compute_arm_targets(dummy_obs)
            return il_targets
            
        except (RuntimeError, ValueError, AttributeError) as e:
            print(f"[WARNING] ARM IL computation failed: {e}. Using default pose.")
            return self._get_default_pose(group_size)
    
    def _compute_il_hand_targets(self, group_size: int) -> torch.Tensor:
        """Compute IL targets for hand joints.
        
        Args:
            group_size: Number of joints in the group
            
        Returns:
            Target joint positions for hand joints (num_envs, group_size)
        """
        try:
            # Get current observations for IL model
            dummy_obs = {"dummy": torch.zeros(self.num_envs, 1, device=self.device)}
            
            il_targets = self._il_controller.compute_hand_targets(dummy_obs)
            return il_targets
            
        except (RuntimeError, ValueError, AttributeError) as e:
            print(f"[WARNING] HAND IL computation failed: {e}. Using default pose.")
            return self._get_default_pose(group_size)
    
    def _get_default_pose(self, group_size: int) -> torch.Tensor:
        """Get default pose for a joint group.
        
        Args:
            group_size: Number of joints in the group
            
        Returns:
            Default joint positions (num_envs, group_size)
        """
        return torch.zeros(self.num_envs, group_size, device=self.device)
    

    
    def _extract_rl_targets_for_group(self, group: JointGroup) -> torch.Tensor:
        """Extract RL targets for a specific group from processed RL actions.
        
        Args:
            group: Joint group to extract targets for
            
        Returns:
            Target joint positions for the specified group
        """
        # Find the indices of this group within the RL-controlled joints
        group_indices = self._joint_group_indices[group]
        
        # Convert to tensor if needed
        if isinstance(group_indices, list):
            group_indices_tensor = torch.tensor(group_indices, device=self.device, dtype=torch.long)
        else:
            group_indices_tensor = group_indices
        
        # Find where these indices appear in the RL joint indices
        rl_group_mask = torch.isin(self._rl_joint_indices, group_indices_tensor)
        
        if rl_group_mask.any():
            return self._rl_joint_pos_target[:, rl_group_mask]
        else:
            # Return zeros if this group is not RL-controlled
            if isinstance(group_indices, list):
                group_size = len(group_indices)
            else:
                group_size = group_indices.numel()
            return torch.zeros(self.num_envs, group_size, device=self.device)
    

    


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

    # Pink IK configuration removed - using Differential IK only
    
    # Trajectory generator configuration (for IK policy)
    trajectory_generator_type: str = "circular"
    """Type of trajectory generator for IK policy. Options: 'circular', 'linear', 'custom'."""
    
    trajectory_generator_params: dict | None = None
    """Parameters for the trajectory generator. If None, default parameters will be used."""
    
    # Upper body IL policy configuration
    upper_body_policy_type: str = "separate"
    """Type of upper body IL policy. Options: 'separate' (arm and hand separate), 'unified' (single policy for both)."""
    
    upper_body_policy_model_path: str | None = None
    """Path to the upper body IL policy model file. If None, dummy model will be used."""
