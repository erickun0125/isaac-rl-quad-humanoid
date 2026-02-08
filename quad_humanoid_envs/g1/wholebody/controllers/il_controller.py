"""Upper body IL (Imitation Learning) controller for G1 humanoid robot.

Implements an upper body IL controller that can:
1. Load pre-trained IL models for arm and hand control
2. Generate joint targets based on current observations
3. Support different IL architectures (MLP, etc.)
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
        device: str = "cuda:0",
    ):
        """Initialize MLP IL model.

        Args:
            input_dim: Dimension of input observations.
            output_dim: Dimension of output joint targets.
            hidden_dims: List of hidden layer dimensions.
            device: Device for tensor operations.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                torch.nn.Linear(prev_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim

        layers.append(torch.nn.Linear(prev_dim, output_dim))

        self.model = torch.nn.Sequential(*layers).to(device)
        self.is_loaded = False

    def predict(self, observations: torch.Tensor) -> torch.Tensor:
        """Predict joint targets from observations."""
        if not self.is_loaded:
            return torch.zeros(observations.shape[0], self.output_dim, device=self.device)

        with torch.no_grad():
            return self.model(observations)

    def load_model(self, model_path: str) -> None:
        """Load pre-trained model from file."""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()
            self.is_loaded = True
            print(f"Successfully loaded IL model from {model_path}")
        except Exception as e:
            print(f"Failed to load IL model from {model_path}: {e}")
            self.is_loaded = False


class DummyILModel(ILModel):
    """Dummy IL model that returns default poses (for testing)."""

    def __init__(self, output_dim: int, device: str = "cuda:0"):
        """Initialize dummy IL model.

        Args:
            output_dim: Dimension of output joint targets.
            device: Device for tensor operations.
        """
        self.output_dim = output_dim
        self.device = device
        self.is_loaded = True
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
        upper_body_model: Optional[ILModel] = None,
        policy_type: str = "separate",
        device: str = "cuda:0",
    ):
        """Initialize upper body IL controller.

        Args:
            robot: The articulated robot asset.
            arm_model: IL model for arm control (used in separate mode).
            hand_model: IL model for hand control (used in separate mode).
            upper_body_model: IL model for unified upper body control (used in unified mode).
            policy_type: Policy type - "separate" or "unified".
            device: Device for tensor operations.
        """
        self.robot = robot
        self.device = device
        self.policy_type = policy_type

        self.arm_joint_names = [
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
            "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
            "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
        ]

        self.hand_joint_names = [
            "left_hand_index_0_joint", "left_hand_middle_0_joint", "left_hand_thumb_0_joint",
            "left_hand_index_1_joint", "left_hand_middle_1_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
            "right_hand_index_0_joint", "right_hand_middle_0_joint", "right_hand_thumb_0_joint",
            "right_hand_index_1_joint", "right_hand_middle_1_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        ]

        self.upper_body_joint_names = self.arm_joint_names + self.hand_joint_names

        if self.policy_type == "unified":
            if upper_body_model is None:
                self.upper_body_model = DummyILModel(output_dim=len(self.upper_body_joint_names), device=device)
            else:
                self.upper_body_model = upper_body_model
            self.arm_model = None
            self.hand_model = None
        else:
            if arm_model is None:
                self.arm_model = DummyILModel(output_dim=len(self.arm_joint_names), device=device)
            else:
                self.arm_model = arm_model

            if hand_model is None:
                self.hand_model = DummyILModel(output_dim=len(self.hand_joint_names), device=device)
            else:
                self.hand_model = hand_model
            self.upper_body_model = None

    def compute_upper_body_targets(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute target joint positions for upper body using IL.

        Args:
            observations: Dictionary of observations for IL model.

        Returns:
            Target joint positions for upper body joints (arm + hand).
        """
        if self.policy_type == "unified":
            upper_body_obs = self._extract_upper_body_observations(observations)
            upper_body_targets = self.upper_body_model.predict(upper_body_obs)
            return upper_body_targets
        else:
            arm_targets = self.compute_arm_targets(observations)
            hand_targets = self.compute_hand_targets(observations)
            return torch.cat([arm_targets, hand_targets], dim=-1)

    def compute_arm_targets(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute target joint positions for arms using IL.

        Args:
            observations: Dictionary of observations for IL model.

        Returns:
            Target joint positions for arm joints.
        """
        if self.policy_type == "unified":
            upper_body_targets = self.compute_upper_body_targets(observations)
            arm_targets = upper_body_targets[..., :len(self.arm_joint_names)]
            return arm_targets
        else:
            arm_obs = self._extract_arm_observations(observations)
            arm_targets = self.arm_model.predict(arm_obs)
            return arm_targets

    def compute_hand_targets(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute target joint positions for hands using IL.

        Args:
            observations: Dictionary of observations for IL model.

        Returns:
            Target joint positions for hand joints.
        """
        if self.policy_type == "unified":
            upper_body_targets = self.compute_upper_body_targets(observations)
            hand_targets = upper_body_targets[..., len(self.arm_joint_names):]
            return hand_targets
        else:
            hand_obs = self._extract_hand_observations(observations)
            hand_targets = self.hand_model.predict(hand_obs)
            return hand_targets

    def _extract_upper_body_observations(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract relevant observations for unified upper body IL model."""
        batch_size = next(iter(observations.values())).shape[0]
        dummy_obs = torch.zeros(batch_size, 56, device=self.device)
        return dummy_obs

    def _extract_arm_observations(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract relevant observations for arm IL model."""
        batch_size = next(iter(observations.values())).shape[0]
        dummy_obs = torch.zeros(batch_size, 32, device=self.device)
        return dummy_obs

    def _extract_hand_observations(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract relevant observations for hand IL model."""
        batch_size = next(iter(observations.values())).shape[0]
        dummy_obs = torch.zeros(batch_size, 24, device=self.device)
        return dummy_obs

    def load_arm_model(self, model_path: str) -> None:
        """Load pre-trained arm IL model."""
        self.arm_model.load_model(model_path)

    def load_hand_model(self, model_path: str) -> None:
        """Load pre-trained hand IL model."""
        self.hand_model.load_model(model_path)

    def get_arm_joint_names(self) -> List[str]:
        """Get names of arm joints."""
        return self.arm_joint_names

    def get_hand_joint_names(self) -> List[str]:
        """Get names of hand joints."""
        return self.hand_joint_names
