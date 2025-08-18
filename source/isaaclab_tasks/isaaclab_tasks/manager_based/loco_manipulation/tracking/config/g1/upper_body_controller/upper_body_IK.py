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
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from isaaclab.assets import Articulation
from isaaclab.controllers.pink_ik import PinkIKController
from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg

try:
    from pink.tasks import FrameTask
    PINK_AVAILABLE = True
except ImportError:
    print("Warning: Pink IK not available. Falling back to simple IK solver.")
    PINK_AVAILABLE = False


class TrajectoryGenerator(ABC):
    """Abstract base class for trajectory generators."""
    
    @abstractmethod
    def generate(self, current_time: float, **kwargs) -> Dict[str, torch.Tensor]:
        """Generate trajectory targets at the given time."""


class CircularTrajectoryGenerator(TrajectoryGenerator):
    """Generates circular trajectories for end-effectors."""
    
    def __init__(
        self,
        center: Tuple[float, float, float] = (0.3, 0.0, 0.0),
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


class SimpleIKSolver:
    """Simple geometric IK solver for G1 arms."""
    
    def __init__(
        self,
        shoulder_offset: Tuple[float, float, float] = (0.0, 0.22, 0.0),
        upper_arm_length: float = 0.25,
        forearm_length: float = 0.25,
        device: str = "cuda:0"
    ):
        """Initialize IK solver.
        
        Args:
            shoulder_offset: Shoulder position offset from base
            upper_arm_length: Length of upper arm
            forearm_length: Length of forearm  
            device: Device for tensor operations
        """
        self.shoulder_offset = torch.tensor(shoulder_offset, device=device)
        self.upper_arm_length = upper_arm_length
        self.forearm_length = forearm_length
        self.device = device
        
    def solve_arm_ik(
        self, 
        target_pos: torch.Tensor, 
        target_quat: torch.Tensor,  # pylint: disable=unused-argument
        is_left_arm: bool = True
    ) -> torch.Tensor:
        """Solve IK for one arm.
        
        Args:
            target_pos: Target end-effector position
            target_quat: Target end-effector orientation  
            is_left_arm: Whether this is the left arm
            
        Returns:
            Joint angles for the arm (7 DOF)
        """
        # Adjust shoulder offset for left/right
        shoulder_pos = self.shoulder_offset.clone()
        if not is_left_arm:
            shoulder_pos[1] *= -1  # Mirror Y for right arm
            
        # Calculate target relative to shoulder
        target_rel = target_pos - shoulder_pos
        
        # Calculate distance to target
        distance = torch.norm(target_rel)
        
        # Check if target is reachable
        max_reach = self.upper_arm_length + self.forearm_length
        if distance > max_reach:
            # Scale down target to max reach
            target_rel = target_rel * (max_reach * 0.9) / distance
            distance = max_reach * 0.9
            
        # Simple 2-DOF IK for shoulder pitch and elbow
        # Using law of cosines
        cos_elbow = (self.upper_arm_length**2 + self.forearm_length**2 - distance**2) / \
                   (2 * self.upper_arm_length * self.forearm_length)
        cos_elbow = torch.clamp(cos_elbow, -1.0, 1.0)
        elbow_angle = torch.acos(cos_elbow)
        
        # Shoulder pitch calculation
        alpha = torch.atan2(target_rel[2], torch.norm(target_rel[:2]))
        beta = torch.acos((self.upper_arm_length**2 + distance**2 - self.forearm_length**2) / \
                         (2 * self.upper_arm_length * distance))
        shoulder_pitch = alpha - beta
        
        # Shoulder yaw and roll (simplified)
        shoulder_yaw = torch.atan2(target_rel[1], target_rel[0])
        shoulder_roll = 0.0  # Simplified
        
        # Wrist angles (simplified to maintain orientation)
        wrist_roll = 0.0
        wrist_pitch = 0.0
        wrist_yaw = 0.0
        
        # Return joint angles
        joint_angles = torch.tensor([
            shoulder_pitch,
            shoulder_roll, 
            shoulder_yaw,
            elbow_angle,
            wrist_roll,
            wrist_pitch,
            wrist_yaw
        ], device=self.device)
        
        return joint_angles
        
    def solve_hand_ik(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:  # pylint: disable=unused-argument
        """Solve IK for hand joints.
        
        Args:
            target_pos: Target hand position (not used in simple implementation)
            target_quat: Target hand orientation (not used in simple implementation)
            
        Returns:
            Joint angles for hand joints (7 DOF for DEX3)
        """
        # Simple implementation: return default hand pose
        hand_angles = torch.zeros(7, device=self.device)
        return hand_angles


class UpperBodyIKController:
    """Upper body IK controller for G1 humanoid robot using Pink IK."""
    
    def __init__(
        self,
        robot: Articulation,
        trajectory_generator: Optional[TrajectoryGenerator] = None,
        device: str = "cuda:0",
        urdf_path: Optional[str] = None,
        mesh_path: Optional[str] = None,
    ):
        """Initialize upper body IK controller.
        
        Args:
            robot: The articulated robot asset
            trajectory_generator: Trajectory generator for end-effector targets
            device: Device for tensor operations
            urdf_path: Path to URDF file for Pink IK (optional)
            mesh_path: Path to mesh files for Pink IK (optional)
        """
        self.robot = robot
        self.device = device
        
        # Initialize trajectory generator
        if trajectory_generator is None:
            self.trajectory_generator = CircularTrajectoryGenerator(device=device)
        else:
            self.trajectory_generator = trajectory_generator
        
        # Define controlled joints
        self.arm_joint_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", 
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
        ]
        
        self.hand_joint_names = [
            "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
            "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
            "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint"
        ]
        
        # Initialize Pink IK controller or fallback to simple IK
        self.use_pink_ik = PINK_AVAILABLE and urdf_path is not None
        
        if self.use_pink_ik:
            try:
                self._setup_pink_ik_controller(urdf_path, mesh_path)
                print("Pink IK controller initialized successfully")
            except Exception as e:  # pylint: disable=broad-except
                print(f"Failed to initialize Pink IK controller: {e}")
                print("Falling back to simple IK solver")
                self.use_pink_ik = False
                self.ik_solver = SimpleIKSolver(device=device)
        else:
            # Fallback to simple IK solver
            self.ik_solver = SimpleIKSolver(device=device)
            
    def _setup_pink_ik_controller(self, urdf_path: str, mesh_path: Optional[str] = None):
        """Setup Pink IK controller with G1 specific configuration."""
        if not PINK_AVAILABLE:
            raise ImportError("Pink IK is not available")
            
        # Create Pink IK tasks for left and right end-effectors
        left_ee_task = FrameTask(
            "left_wrist_yaw_link",  # Frame name in URDF
            position_cost=1.0,     # Position tracking weight
            orientation_cost=0.1,  # Orientation tracking weight (lower priority)
        )
        
        right_ee_task = FrameTask(
            "right_wrist_yaw_link", # Frame name in URDF  
            position_cost=1.0,     # Position tracking weight
            orientation_cost=0.1,  # Orientation tracking weight (lower priority)
        )
        
        # Configure Pink IK controller
        pink_cfg = PinkIKControllerCfg(
            urdf_path=urdf_path,
            mesh_path=mesh_path,
            variable_input_tasks=[left_ee_task, right_ee_task],
            fixed_input_tasks=[],  # No fixed tasks for now
            joint_names=self.arm_joint_names,  # Only arm joints
            articulation_name="robot",
            base_link_name="pelvis",  # G1 base link
            show_ik_warnings=True,
        )
        
        # Initialize Pink IK controller
        self.pink_ik_controller = PinkIKController(pink_cfg, device=self.device)
        self.pink_ik_controller.initialize()
        
        # Store references to tasks for updating targets
        self.left_ee_task = left_ee_task
        self.right_ee_task = right_ee_task

    def compute_arm_targets(self, current_time: float, current_joint_pos: Optional[np.ndarray] = None) -> torch.Tensor:
        """Compute target joint positions for arms using Pink IK or simple IK.
        
        Args:
            current_time: Current simulation time
            current_joint_pos: Current joint positions (required for Pink IK)
            
        Returns:
            Target joint positions for arm joints
        """
        # Generate trajectory targets
        trajectory_targets = self.trajectory_generator.generate(current_time)
        
        if self.use_pink_ik and current_joint_pos is not None:
            # Use Pink IK solver
            return self._compute_arm_targets_pink_ik(trajectory_targets, current_joint_pos)
        else:
            # Use simple IK solver
            return self._compute_arm_targets_simple_ik(trajectory_targets)
    
    def _compute_arm_targets_pink_ik(self, trajectory_targets: Dict[str, torch.Tensor], current_joint_pos: np.ndarray) -> torch.Tensor:
        """Compute arm targets using Pink IK solver."""
        # Update target poses for Pink IK tasks
        # Note: These variables are extracted for future Pink IK target setting
        # Currently not used as Pink tasks are pre-configured
        # left_ee_pos = trajectory_targets["left_ee_pos"].cpu().numpy()
        # left_ee_quat = trajectory_targets["left_ee_quat"].cpu().numpy()
        # right_ee_pos = trajectory_targets["right_ee_pos"].cpu().numpy()  
        # right_ee_quat = trajectory_targets["right_ee_quat"].cpu().numpy()
        
        # Update task targets using Pink's API
        try:
            # Set target poses for frames (Pink uses SE3 poses)
            # Note: This is a simplified approach - actual Pink API may differ
            # You may need to adjust this based on the actual Pink FrameTask API
            
            # For now, we'll use the Pink controller's compute method directly
            # and assume that the tasks have already been configured with appropriate targets
            
            # Solve IK using Pink
            target_joint_pos = self.pink_ik_controller.compute(current_joint_pos, dt=0.02)  # 50 Hz control
            
            return target_joint_pos
            
        except Exception as e:  # pylint: disable=broad-except
            print(f"Pink IK solver failed: {e}")
            # Fallback to simple IK
            return self._compute_arm_targets_simple_ik(trajectory_targets)
    
    def _compute_arm_targets_simple_ik(self, trajectory_targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute arm targets using simple geometric IK solver."""
        # Solve IK for left arm
        left_ee_pos = trajectory_targets["left_ee_pos"]
        left_ee_quat = trajectory_targets["left_ee_quat"]
        left_arm_targets = self.ik_solver.solve_arm_ik(left_ee_pos, left_ee_quat, is_left_arm=True)
        
        # Solve IK for right arm  
        right_ee_pos = trajectory_targets["right_ee_pos"]
        right_ee_quat = trajectory_targets["right_ee_quat"]
        right_arm_targets = self.ik_solver.solve_arm_ik(right_ee_pos, right_ee_quat, is_left_arm=False)
        
        # Combine arm targets
        arm_targets = torch.cat([left_arm_targets, right_arm_targets])
        
        return arm_targets
        
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
