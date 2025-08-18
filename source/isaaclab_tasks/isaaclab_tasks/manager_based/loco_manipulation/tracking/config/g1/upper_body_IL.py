# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Upper body IL (Imitation Learning) controller module for G1 humanoid robot.

This module implements an upper body IL controller that can:
1. Load pre-trained IL models for arm and hand control
2. Generate joint targets based on current observations
3. Support different IL architectures (MLP, LSTM, Transformer, etc.)

The controller operates independently from the RL policy and provides
IL-based joint targets for upper body control.
"""

import torch
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

from isaaclab.assets import Articulation


class ILModel(ABC):
    """Abstract base class for IL models."""
    
    @abstractmethod
    def predict(self, observations: torch.Tensor) -> torch.Tensor:
        """Predict joint targets from observations."""
        
    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load pre-trained model from file."""


class MLPILModel(ILModel):
    """MLP-based IL model for joint prediction."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Optional[List[int]] = None,
        device: str = "cuda:0"
    ):
        """Initialize MLP IL model.
        
        Args:
            input_dim: Dimension of input observations
            output_dim: Dimension of output joint targets
            hidden_dims: List of hidden layer dimensions
            device: Device for tensor operations
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        
        # Set default hidden dimensions if none provided
        if hidden_dims is None:
            hidden_dims = [256, 128]
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
            
        layers.append(torch.nn.Linear(prev_dim, output_dim))
        
        self.model = torch.nn.Sequential(*layers).to(device)
        self.is_loaded = False
        
    def predict(self, observations: torch.Tensor) -> torch.Tensor:
        """Predict joint targets from observations."""
        if not self.is_loaded:
            # Return default pose if model not loaded
            return torch.zeros(observations.shape[0], self.output_dim, device=self.device)
            
        with torch.no_grad():
            return self.model(observations)
            
    def load_model(self, model_path: str) -> None:
        """Load pre-trained model from file."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.is_loaded = True
            print(f"Successfully loaded IL model from {model_path}")
        except Exception as e:  # pylint: disable=broad-except
            print(f"Failed to load IL model from {model_path}: {e}")
            self.is_loaded = False


class DummyILModel(ILModel):
    """Dummy IL model that returns default poses (for testing)."""
    
    def __init__(self, output_dim: int, device: str = "cuda:0"):
        """Initialize dummy IL model.
        
        Args:
            output_dim: Dimension of output joint targets
            device: Device for tensor operations
        """
        self.output_dim = output_dim
        self.device = device
        self.is_loaded = True
        
        # Default pose for arms/hands
        self.default_pose = torch.zeros(output_dim, device=device)
        
    def predict(self, observations: torch.Tensor) -> torch.Tensor:
        """Return default pose."""
        batch_size = observations.shape[0]
        return self.default_pose.unsqueeze(0).expand(batch_size, -1)
        
    def load_model(self, model_path: str) -> None:
        """No-op for dummy model."""
        print(f"Dummy IL model - ignoring model path: {model_path}")


class UpperBodyILController:
    """Upper body IL controller for G1 humanoid robot."""
    
    def __init__(
        self,
        robot: Articulation,
        arm_model: Optional[ILModel] = None,
        hand_model: Optional[ILModel] = None,
        device: str = "cuda:0",
    ):
        """Initialize upper body IL controller.
        
        Args:
            robot: The articulated robot asset
            arm_model: IL model for arm control
            hand_model: IL model for hand control
            device: Device for tensor operations
        """
        self.robot = robot
        self.device = device
        
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
        
        # Initialize IL models
        if arm_model is None:
            self.arm_model = DummyILModel(output_dim=len(self.arm_joint_names), device=device)
        else:
            self.arm_model = arm_model
            
        if hand_model is None:
            self.hand_model = DummyILModel(output_dim=len(self.hand_joint_names), device=device)
        else:
            self.hand_model = hand_model

    def compute_arm_targets(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute target joint positions for arms using IL.
        
        Args:
            observations: Dictionary of observations for IL model
            
        Returns:
            Target joint positions for arm joints
        """
        # Extract relevant observations for arm IL model
        # This would typically include base state, current joint positions, etc.
        arm_obs = self._extract_arm_observations(observations)
        
        # Predict arm targets using IL model
        arm_targets = self.arm_model.predict(arm_obs)
        
        return arm_targets
        
    def compute_hand_targets(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute target joint positions for hands using IL.
        
        Args:
            observations: Dictionary of observations for IL model
            
        Returns:
            Target joint positions for hand joints
        """
        # Extract relevant observations for hand IL model
        hand_obs = self._extract_hand_observations(observations)
        
        # Predict hand targets using IL model
        hand_targets = self.hand_model.predict(hand_obs)
        
        return hand_targets
        
    def _extract_arm_observations(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract relevant observations for arm IL model.
        
        Args:
            observations: Full observation dictionary
            
        Returns:
            Concatenated observations for arm IL model
        """
        # For now, return a dummy observation vector
        # In practice, this would extract relevant features like:
        # - Base orientation
        # - Current arm joint positions/velocities
        # - Target end-effector pose
        # - etc.
        
        batch_size = next(iter(observations.values())).shape[0]
        dummy_obs = torch.zeros(batch_size, 32, device=self.device)  # 32-dim dummy observation
        return dummy_obs
        
    def _extract_hand_observations(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract relevant observations for hand IL model.
        
        Args:
            observations: Full observation dictionary
            
        Returns:
            Concatenated observations for hand IL model
        """
        # Similar to arm observations, extract relevant features for hand control
        batch_size = next(iter(observations.values())).shape[0]
        dummy_obs = torch.zeros(batch_size, 24, device=self.device)  # 24-dim dummy observation
        return dummy_obs
        
    def load_arm_model(self, model_path: str) -> None:
        """Load pre-trained arm IL model.
        
        Args:
            model_path: Path to the arm IL model file
        """
        self.arm_model.load_model(model_path)
        
    def load_hand_model(self, model_path: str) -> None:
        """Load pre-trained hand IL model.
        
        Args:
            model_path: Path to the hand IL model file
        """
        self.hand_model.load_model(model_path)
        
    def get_arm_joint_names(self) -> List[str]:
        """Get names of arm joints."""
        return self.arm_joint_names
        
    def get_hand_joint_names(self) -> List[str]:
        """Get names of hand joints."""
        return self.hand_joint_names
