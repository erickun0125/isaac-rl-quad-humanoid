"""Upper body IK controller for G1 humanoid robot.

Implements an upper body IK controller consisting of:
1. Cartesian trajectory generator for end-effector targets
2. Pink IK solver for arm control (with simple IK fallback)
3. Simple hand joint control (set to zeros)
"""

import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from isaaclab.assets import Articulation

try:
    from isaaclab.controllers.pink_ik import PinkIKController
    from isaaclab.controllers.pink_ik_cfg import PinkIKControllerCfg
    from pink.tasks import FrameTask
    PINK_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Pink IK not available ({e}). Falling back to simple IK.")
    PINK_AVAILABLE = False
    PinkIKController = None
    PinkIKControllerCfg = None
    FrameTask = None


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
        device: str = "cuda:0",
    ):
        """Initialize circular trajectory generator.

        Args:
            center: Center of the circular trajectory in robot base frame.
            radius: Radius of the circular trajectory.
            frequency: Frequency of the circular motion (Hz).
            device: Device for tensor operations.
        """
        self.center = torch.tensor(center, device=device)
        self.radius = radius
        self.frequency = frequency
        self.device = device

    def generate(self, current_time: float, **kwargs) -> Dict[str, torch.Tensor]:
        """Generate circular trajectory targets."""
        phase = 2 * math.pi * self.frequency * current_time

        y_offset = self.radius * torch.cos(torch.tensor(phase, device=self.device))
        z_offset = self.radius * torch.sin(torch.tensor(phase, device=self.device))

        left_target = self.center.clone()
        left_target[1] += y_offset + 0.2
        left_target[2] += z_offset

        right_target = self.center.clone()
        right_target[1] -= y_offset + 0.2
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
        device: str = "cuda:0",
    ):
        """Initialize IK solver.

        Args:
            shoulder_offset: Shoulder position offset from base.
            upper_arm_length: Length of upper arm.
            forearm_length: Length of forearm.
            device: Device for tensor operations.
        """
        self.shoulder_offset = torch.tensor(shoulder_offset, device=device)
        self.upper_arm_length = upper_arm_length
        self.forearm_length = forearm_length
        self.device = device

    def solve_arm_ik(
        self,
        target_pos: torch.Tensor,
        target_quat: torch.Tensor,
        is_left_arm: bool = True,
    ) -> torch.Tensor:
        """Solve IK for one arm.

        Args:
            target_pos: Target end-effector position.
            target_quat: Target end-effector orientation (unused in simple solver).
            is_left_arm: Whether this is the left arm.

        Returns:
            Joint angles for the arm (7 DOF).
        """
        shoulder_pos = self.shoulder_offset.clone()
        if not is_left_arm:
            shoulder_pos[1] *= -1

        target_rel = target_pos - shoulder_pos
        distance = torch.norm(target_rel)

        max_reach = self.upper_arm_length + self.forearm_length
        if distance > max_reach:
            target_rel = target_rel * (max_reach * 0.9) / distance
            distance = max_reach * 0.9

        cos_elbow = (self.upper_arm_length**2 + self.forearm_length**2 - distance**2) / (
            2 * self.upper_arm_length * self.forearm_length
        )
        cos_elbow = torch.clamp(torch.tensor(cos_elbow, device=self.device), -1.0, 1.0)
        elbow_angle = torch.acos(cos_elbow)

        alpha = torch.atan2(target_rel[2], torch.norm(target_rel[:2]))
        cos_beta = (self.upper_arm_length**2 + distance**2 - self.forearm_length**2) / (
            2 * self.upper_arm_length * distance
        )
        cos_beta = torch.clamp(torch.tensor(cos_beta, device=self.device), -1.0, 1.0)
        beta = torch.acos(cos_beta)
        shoulder_pitch = alpha - beta

        shoulder_yaw = torch.atan2(target_rel[1], target_rel[0])
        shoulder_roll = torch.tensor(0.0, device=self.device)

        wrist_roll = torch.tensor(0.0, device=self.device)
        wrist_pitch = torch.tensor(0.0, device=self.device)
        wrist_yaw = torch.tensor(0.0, device=self.device)

        joint_angles = torch.tensor(
            [
                shoulder_pitch,
                shoulder_roll,
                shoulder_yaw,
                elbow_angle,
                wrist_roll,
                wrist_pitch,
                wrist_yaw,
            ],
            device=self.device,
        )

        return joint_angles

    def solve_hand_ik(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """Solve IK for hand joints (returns default zeros).

        Args:
            target_pos: Target hand position (unused in simple implementation).
            target_quat: Target hand orientation (unused in simple implementation).

        Returns:
            Joint angles for hand joints (7 DOF for DEX3).
        """
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
            robot: The articulated robot asset.
            trajectory_generator: Trajectory generator for end-effector targets.
            device: Device for tensor operations.
            urdf_path: Path to URDF file for Pink IK (optional).
            mesh_path: Path to mesh files for Pink IK (optional).
        """
        self.robot = robot
        self.device = device

        if trajectory_generator is None:
            self.trajectory_generator = CircularTrajectoryGenerator(device=device)
        else:
            self.trajectory_generator = trajectory_generator

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

        self.use_pink_ik = PINK_AVAILABLE and urdf_path is not None

        if self.use_pink_ik:
            try:
                self._setup_pink_ik_controller(urdf_path, mesh_path)
                print("Pink IK controller initialized successfully")
            except Exception as e:
                print(f"Failed to initialize Pink IK controller: {e}")
                print("Falling back to simple IK solver")
                self.use_pink_ik = False
                self.ik_solver = SimpleIKSolver(device=device)
        else:
            self.ik_solver = SimpleIKSolver(device=device)

    def _setup_pink_ik_controller(self, urdf_path: str, mesh_path: Optional[str] = None):
        """Setup Pink IK controller with G1 specific configuration."""
        if not PINK_AVAILABLE or FrameTask is None or PinkIKControllerCfg is None:
            raise ImportError("Pink IK is not available")

        left_ee_task = FrameTask(
            "left_wrist_yaw_link",
            position_cost=1.0,
            orientation_cost=0.1,
        )

        right_ee_task = FrameTask(
            "right_wrist_yaw_link",
            position_cost=1.0,
            orientation_cost=0.1,
        )

        pink_cfg = PinkIKControllerCfg(
            urdf_path=urdf_path,
            mesh_path=mesh_path,
            variable_input_tasks=[left_ee_task, right_ee_task],
            fixed_input_tasks=[],
            joint_names=self.arm_joint_names,
            articulation_name="robot",
            base_link_name="pelvis",
            show_ik_warnings=True,
        )

        if PinkIKController is None:
            raise ImportError("PinkIKController is not available")
        self.pink_ik_controller = PinkIKController(pink_cfg, device=self.device)
        self.pink_ik_controller.initialize()

        self.left_ee_task = left_ee_task
        self.right_ee_task = right_ee_task

    def compute_arm_targets(
        self, current_time: float, current_joint_pos: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """Compute target joint positions for arms using Pink IK or simple IK.

        Args:
            current_time: Current simulation time.
            current_joint_pos: Current joint positions (required for Pink IK).

        Returns:
            Target joint positions for arm joints.
        """
        trajectory_targets = self.trajectory_generator.generate(current_time)

        if self.use_pink_ik and current_joint_pos is not None:
            return self._compute_arm_targets_pink_ik(trajectory_targets, current_joint_pos)
        else:
            return self._compute_arm_targets_simple_ik(trajectory_targets)

    def _compute_arm_targets_pink_ik(
        self, trajectory_targets: Dict[str, torch.Tensor], current_joint_pos: np.ndarray
    ) -> torch.Tensor:
        """Compute arm targets using Pink IK solver."""
        try:
            target_joint_pos = self.pink_ik_controller.compute(current_joint_pos, dt=0.02)
            return target_joint_pos
        except Exception as e:
            print(f"Pink IK solver failed: {e}")
            return self._compute_arm_targets_simple_ik(trajectory_targets)

    def _compute_arm_targets_simple_ik(
        self, trajectory_targets: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute arm targets using simple geometric IK solver."""
        left_arm_targets = self.ik_solver.solve_arm_ik(
            trajectory_targets["left_ee_pos"], trajectory_targets["left_ee_quat"], is_left_arm=True
        )

        right_arm_targets = self.ik_solver.solve_arm_ik(
            trajectory_targets["right_ee_pos"], trajectory_targets["right_ee_quat"], is_left_arm=False
        )

        arm_targets = torch.cat([left_arm_targets, right_arm_targets])
        return arm_targets

    def compute_hand_targets(self, current_time: float) -> torch.Tensor:
        """Compute target joint positions for hands (set to zeros).

        Args:
            current_time: Current simulation time (unused for now).

        Returns:
            Target joint positions for hand joints (all zeros).
        """
        num_hand_joints = len(self.hand_joint_names)
        hand_targets = torch.zeros(num_hand_joints, device=self.device)
        return hand_targets

    def get_arm_joint_names(self) -> List[str]:
        """Get names of arm joints."""
        return self.arm_joint_names

    def get_hand_joint_names(self) -> List[str]:
        """Get names of hand joints."""
        return self.hand_joint_names
