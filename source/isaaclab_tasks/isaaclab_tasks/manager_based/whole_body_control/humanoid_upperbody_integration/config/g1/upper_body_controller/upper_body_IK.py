# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Upper body IK controller module for G1 humanoid robot.

This module implements an upper body IK controller that consists of:
1. Cartesian trajectory generator for end-effector targets
2. Pink IK solver for arm control
3. Simple hand joint control (set to zeros)

The controller operates independently from the RL policy and provides
IK-based joint targets for upper body control.
"""

import math
import torch
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from isaaclab.assets import Articulation


class IKDataCollector:
    """Collects all necessary data for IK computation in a modular way."""
    
    def __init__(self, asset: Articulation, joint_group_indices: dict, device: str):
        """Initialize IK data collector.
        
        Args:
            asset: Articulated robot asset
            joint_group_indices: Dictionary mapping joint groups to indices
            device: Computing device
        """
        self.asset = asset
        self.joint_group_indices = joint_group_indices
        self.device = device
        
        # Import JointGroup enum locally to avoid circular imports
        from ..mdp.whole_body_actions import JointGroup
        self.JointGroup = JointGroup
        
        # Cache end-effector body IDs for efficiency
        self._left_ee_body_id = None
        self._right_ee_body_id = None
        self._initialize_ee_body_ids()
    
    def _initialize_ee_body_ids(self):
        """Initialize end-effector body IDs."""
        try:
            left_body_ids = self.asset.find_bodies("left_hand_thumb_0_link")[0]
            if len(left_body_ids) > 0:
                self._left_ee_body_id = left_body_ids[0]
                
            right_body_ids = self.asset.find_bodies("right_hand_thumb_0_link")[0]
            if len(right_body_ids) > 0:
                self._right_ee_body_id = right_body_ids[0]
        except (RuntimeError, IndexError, AttributeError) as e:
            print(f"[WARNING] Failed to initialize end-effector body IDs: {e}")
    
    def get_current_joint_positions(self, group) -> torch.Tensor:
        """Get current joint positions for a specific group.
        
        Args:
            group: Joint group (JointGroup enum)
            
        Returns:
            Current joint positions (num_envs, num_joints_in_group)
        """
        group_indices = self.joint_group_indices[group]
        if len(group_indices) > 0:
            return self.asset.data.joint_pos[:, group_indices]
        else:
            return torch.zeros(self.asset.num_instances, 0, device=self.device)
    
    def get_end_effector_pose(self, side: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get current end-effector pose for the specified side.
        
        Args:
            side: "left" or "right"
            
        Returns:
            Tuple of (position, quaternion) tensors (num_envs, 3) and (num_envs, 4)
        """
        body_id = self._left_ee_body_id if side == "left" else self._right_ee_body_id
        
        if body_id is not None:
            try:
                ee_pos = self.asset.data.body_pos_w[:, body_id]
                ee_quat = self.asset.data.body_quat_w[:, body_id]
                return ee_pos, ee_quat
            except (RuntimeError, IndexError, AttributeError) as e:
                print(f"[WARNING] Failed to get {side} end-effector pose: {e}")
        
        # Return zero pose as fallback
        num_envs = self.asset.num_instances
        return (torch.zeros(num_envs, 3, device=self.device),
                torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=self.device).expand(num_envs, -1))
    
    def get_arm_jacobian(self, side: str) -> torch.Tensor:
        """Get Jacobian matrix for the specified arm.
        
        Args:
            side: "left" or "right"
            
        Returns:
            Jacobian matrix tensor (num_envs, 6, 7) for single arm
        """
        body_id = self._left_ee_body_id if side == "left" else self._right_ee_body_id
        
        if body_id is not None:
            try:
                # Get arm joint indices for this side
                arm_joint_indices = self.joint_group_indices[self.JointGroup.ARM]
                half_joints = len(arm_joint_indices) // 2
                
                if side == "left":
                    side_joint_indices = arm_joint_indices[:half_joints]
                else:
                    side_joint_indices = arm_joint_indices[half_joints:]
                
                # Try to get Jacobian using the root_physx_view API
                ee_jacobi_idx = body_id - 1
                jacobian_w = self.asset.root_physx_view.get_jacobians()
                
                if jacobian_w is not None and jacobian_w.shape[-1] >= len(side_joint_indices):
                    jacobian = jacobian_w[:, ee_jacobi_idx, :6, side_joint_indices]
                    return jacobian
                else:
                    raise RuntimeError("Jacobian computation via physx_view failed")
                    
            except (RuntimeError, IndexError, AttributeError) as e:
                print(f"[WARNING] Jacobian computation failed for {side} arm: {e}")
        
        # Fallback: return identity-like Jacobian
        num_envs = self.asset.num_instances
        return torch.eye(6, 7, device=self.device).unsqueeze(0).expand(num_envs, -1, -1)
    
    def collect_ik_data(self) -> dict:
        """Collect all IK data needed for arm control.
        
        Returns:
            Dictionary containing all IK data
        """
        data = {}
        
        # Get current arm joint positions
        data['current_arm_joints'] = self.get_current_joint_positions(self.JointGroup.ARM)
        
        # Get end-effector poses
        data['left_ee_pos'], data['left_ee_quat'] = self.get_end_effector_pose("left")
        data['right_ee_pos'], data['right_ee_quat'] = self.get_end_effector_pose("right")
        
        # Get Jacobians
        data['left_jacobian'] = self.get_arm_jacobian("left")
        data['right_jacobian'] = self.get_arm_jacobian("right")
        
        return data


# Import Differential IK components
from isaaclab.controllers.differential_ik import DifferentialIKController
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg


class TrajectoryGenerator(ABC):
    """Abstract base class for trajectory generators."""
    
    @abstractmethod
    def generate(self, current_time: float, **kwargs) -> Dict[str, torch.Tensor]:
        """Generate trajectory targets at the given time."""


class CircularTrajectoryGenerator(TrajectoryGenerator):
    """Generates circular trajectories for end-effectors."""
    
    def __init__(
        self,
        center: Tuple[float, float, float] = (0.3, 0.0, 0.3),
        radius: float = 0.1,
        frequency: float = 0.5,
        device: str = "cuda:0"
    ):
        """Initialize circular trajectory generator.
        
        Args:
            center: Center of the circular trajectory in robot base frame
            radius: Radius of the circular trajectory
            frequency: Frequency of the circular motion (Hz)
            device: Device for tensor operations
        """
        self.center = torch.tensor(center, device=device)
        self.radius = radius
        self.frequency = frequency
        self.device = device
        
    def generate(self, current_time: float, **kwargs) -> Dict[str, torch.Tensor]:
        """Generate circular trajectory targets."""
        # Calculate phase
        phase = 2 * math.pi * self.frequency * current_time
        
        # Generate circular motion in YZ plane
        y_offset = self.radius * torch.cos(torch.tensor(phase, device=self.device))
        z_offset = self.radius * torch.sin(torch.tensor(phase, device=self.device))
        
        # Left and right hand targets (mirrored in Y)
        left_target = self.center.clone()
        left_target[1] += y_offset + 0.2  # Offset for left side
        left_target[2] += z_offset
        
        right_target = self.center.clone()
        right_target[1] -= y_offset + 0.2  # Offset for right side (mirrored)
        right_target[2] += z_offset
        
        return {
            "left_ee_pos": left_target,
            "right_ee_pos": right_target,
            "left_ee_quat": torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device),
            "right_ee_quat": torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device),
        }


class DifferentialIKSolver:
    """Differential IK solver wrapper for G1 arms using IsaacLab's DifferentialIKController."""
    
    def __init__(
        self,
        robot: Articulation,
        num_envs: int = 1,
        device: str = "cuda:0"
    ):
        """Initialize differential IK solver.
        
        Args:
            robot: The articulated robot asset
            num_envs: Number of environments
            device: Device for tensor operations
        """
        self.robot = robot
        self.num_envs = num_envs
        self.device = device
        
        # Create separate IK controllers for left and right arms
        self._setup_arm_controllers()
        
    def _setup_arm_controllers(self):
        """Setup differential IK controllers for left and right arms."""
        # Configuration for left arm IK controller
        left_arm_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",  # Damped least squares for better stability
            ik_params={"lambda_val": 0.01}
        )
        
        # Configuration for right arm IK controller  
        right_arm_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",  # Damped least squares for better stability
            ik_params={"lambda_val": 0.01}
        )
        
        # Initialize IK controllers
        self.left_arm_ik = DifferentialIKController(left_arm_cfg, self.num_envs, self.device)
        self.right_arm_ik = DifferentialIKController(right_arm_cfg, self.num_envs, self.device)
        
    def solve_arm_ik(
        self, 
        target_pos: torch.Tensor, 
        target_quat: torch.Tensor,
        is_left_arm: bool = True,
        current_joint_pos: Optional[torch.Tensor] = None,
        jacobian: Optional[torch.Tensor] = None,
        ee_pos: Optional[torch.Tensor] = None,
        ee_quat: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Solve IK for one arm using differential IK.
        
        Args:
            target_pos: Target end-effector position (batch_size, 3)
            target_quat: Target end-effector orientation (batch_size, 4)
            is_left_arm: Whether this is the left arm
            current_joint_pos: Current joint positions (batch_size, 7)
            jacobian: Current Jacobian matrix (batch_size, 6, 7)
            ee_pos: Current end-effector position (batch_size, 3)
            ee_quat: Current end-effector orientation (batch_size, 4)
            
        Returns:
            Joint angles for the arm (batch_size, 7)
        """
        # Check if required inputs are provided
        if (current_joint_pos is None or jacobian is None or 
            ee_pos is None or ee_quat is None):
            print(f"[WARNING] Required inputs missing for {'left' if is_left_arm else 'right'} arm IK")
            batch_size = target_pos.shape[0] if target_pos.dim() > 1 else 1
            return torch.zeros(batch_size, 7, device=self.device)
        
        # Create pose command (position + quaternion)
        pose_command = torch.cat([target_pos, target_quat], dim=-1)
        
        # Select appropriate IK controller
        ik_controller = self.left_arm_ik if is_left_arm else self.right_arm_ik
        
        # Set command for the IK controller
        ik_controller.set_command(pose_command, ee_pos, ee_quat)
        
        # Validate Jacobian shape: expected (batch_size, 6, 7)
        if jacobian.shape[-2] != 6 or jacobian.shape[-1] != 7:
            print(f"[WARNING] Invalid Jacobian shape: {jacobian.shape}. Expected (batch, 6, 7)")
            batch_size = current_joint_pos.shape[0]
            jacobian = torch.zeros(batch_size, 6, 7, device=self.device)
        
        # Compute target joint positions
        try:
            target_joint_pos = ik_controller.compute(ee_pos, ee_quat, jacobian, current_joint_pos)
            return target_joint_pos
        except (RuntimeError, ValueError, AttributeError) as e:
            print(f"[ERROR] IK computation failed for {'left' if is_left_arm else 'right'} arm: {e}")
            batch_size = current_joint_pos.shape[0]
            return torch.zeros(batch_size, 7, device=self.device)
        
    def solve_hand_ik(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:  # pylint: disable=unused-argument
        """Solve IK for hand joints.
        
        Args:
            target_pos: Target hand position (not used in this implementation)
            target_quat: Target hand orientation (not used in this implementation)
            
        Returns:
            Joint angles for hand joints (7 DOF for DEX3)
        """
        # Simple implementation: return default hand pose
        hand_angles = torch.zeros(7, device=self.device)
        return hand_angles


class UpperBodyIKController:
    """Upper body IK controller for G1 humanoid robot using Differential IK."""
    
    def __init__(
        self,
        robot: Articulation,
        trajectory_generator: Optional[TrajectoryGenerator] = None,
        device: str = "cuda:0",
        num_envs: int = 1,
    ):
        """Initialize upper body IK controller.
        
        Args:
            robot: The articulated robot asset
            trajectory_generator: Trajectory generator for end-effector targets
            device: Device for tensor operations
            num_envs: Number of environments
        """
        self.robot = robot
        self.device = device
        self.num_envs = num_envs
        
        # Initialize trajectory generator
        if trajectory_generator is None:
            self.trajectory_generator = CircularTrajectoryGenerator(device=device)
        else:
            self.trajectory_generator = trajectory_generator
        
        # Define controlled joints - 각 팔당 7DOF
        self.left_arm_joint_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"
        ]
        
        self.right_arm_joint_names = [
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", 
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]
        
        self.arm_joint_names = self.left_arm_joint_names + self.right_arm_joint_names
        
        self.hand_joint_names = [
            "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
            "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
            "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"
        ]
        
        # Initialize Differential IK solver
        print(f"[INFO] Initializing Differential IK solver for {num_envs} environments")
        self.ik_solver = DifferentialIKSolver(robot=robot, num_envs=num_envs, device=device)
            


    def compute_arm_targets(
        self, 
        current_time: float, 
        current_joint_pos: Optional[torch.Tensor] = None,
        left_ee_pos: Optional[torch.Tensor] = None,
        left_ee_quat: Optional[torch.Tensor] = None,
        right_ee_pos: Optional[torch.Tensor] = None,
        right_ee_quat: Optional[torch.Tensor] = None,
        left_jacobian: Optional[torch.Tensor] = None,
        right_jacobian: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute target joint positions for arms using Differential IK.
        
        Args:
            current_time: Current simulation time
            current_joint_pos: Current joint positions (num_envs, 14) - 14 arm joints
            left_ee_pos: Current left end-effector position (num_envs, 3)
            left_ee_quat: Current left end-effector orientation (num_envs, 4)
            right_ee_pos: Current right end-effector position (num_envs, 3)
            right_ee_quat: Current right end-effector orientation (num_envs, 4)
            left_jacobian: Current left arm Jacobian matrix (num_envs, 6, 7)
            right_jacobian: Current right arm Jacobian matrix (num_envs, 6, 7)
            
        Returns:
            Target joint positions for arm joints (num_envs, 14)
        """
        # Generate trajectory targets
        trajectory_targets = self.trajectory_generator.generate(current_time)
        
        # Extract target poses
        left_target_pos = trajectory_targets["left_ee_pos"]  # (3,)
        left_target_quat = trajectory_targets["left_ee_quat"]  # (4,)
        right_target_pos = trajectory_targets["right_ee_pos"]  # (3,)
        right_target_quat = trajectory_targets["right_ee_quat"]  # (4,)
        
        # Check if required inputs are available for differential IK
        if (current_joint_pos is None or left_ee_pos is None or right_ee_pos is None or 
            left_jacobian is None or right_jacobian is None):
            print("[WARNING] Required inputs for differential IK missing. Using default pose.")
            return self._get_default_arm_pose()
        
        # Ensure correct tensor shapes
        if current_joint_pos.dim() == 1:
            current_joint_pos = current_joint_pos.unsqueeze(0)
        
        batch_size = current_joint_pos.shape[0]
        
        # Split arm joint positions: left (first 7) and right (last 7) 
        left_current_joints = current_joint_pos[:, :7]  # (batch_size, 7)
        right_current_joints = current_joint_pos[:, 7:14]  # (batch_size, 7)
        
        # Expand target poses to batch size
        left_target_pos_batch = left_target_pos.unsqueeze(0).expand(batch_size, -1)
        left_target_quat_batch = left_target_quat.unsqueeze(0).expand(batch_size, -1)
        right_target_pos_batch = right_target_pos.unsqueeze(0).expand(batch_size, -1)
        right_target_quat_batch = right_target_quat.unsqueeze(0).expand(batch_size, -1)
        
        # Compute IK for left arm
        left_arm_targets = self.ik_solver.solve_arm_ik(
            target_pos=left_target_pos_batch,
            target_quat=left_target_quat_batch,
            is_left_arm=True,
            current_joint_pos=left_current_joints,
            jacobian=left_jacobian,
            ee_pos=left_ee_pos,
            ee_quat=left_ee_quat
        )
        
        # Compute IK for right arm
        right_arm_targets = self.ik_solver.solve_arm_ik(
            target_pos=right_target_pos_batch,
            target_quat=right_target_quat_batch,
            is_left_arm=False,
            current_joint_pos=right_current_joints,
            jacobian=right_jacobian,
            ee_pos=right_ee_pos,
            ee_quat=right_ee_quat
        )
        
        # Combine left and right arm targets
        arm_targets = torch.cat([left_arm_targets, right_arm_targets], dim=-1)
        
        return arm_targets
    

    def _get_default_arm_pose(self) -> torch.Tensor:
        """Get default arm pose when IK fails.
        
        Returns:
            Default arm joint positions (num_envs, 14)
        """
        # Default arm pose: slightly bent arms
        default_left_arm = torch.tensor([
            0.0, 0.0, 0.0,    # shoulder joints
            -0.5,             # elbow (bent)
            0.0, 0.0, 0.0     # wrist joints
        ], device=self.device)
        
        default_right_arm = torch.tensor([
            0.0, 0.0, 0.0,    # shoulder joints  
            -0.5,             # elbow (bent)
            0.0, 0.0, 0.0     # wrist joints
        ], device=self.device)
        
        default_pose = torch.cat([default_left_arm, default_right_arm])
        return default_pose.unsqueeze(0).expand(self.num_envs, -1)
        
    def compute_hand_targets(self, current_time: float) -> torch.Tensor:  # pylint: disable=unused-argument
        """Compute target joint positions for hands (set to zeros).
        
        Args:
            current_time: Current simulation time (unused for now)
            
        Returns:
            Target joint positions for hand joints (all zeros)
        """
        # For now, set all hand joints to zero as requested
        # This creates a tensor with zeros for all hand joints (14 DOF total)
        num_hand_joints = len(self.hand_joint_names)
        hand_targets = torch.zeros(num_hand_joints, device=self.device)
        
        return hand_targets
        
    def get_arm_joint_names(self) -> List[str]:
        """Get names of arm joints."""
        return self.arm_joint_names
        
    def get_hand_joint_names(self) -> List[str]:
        """Get names of hand joints."""
        return self.hand_joint_names
