# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint of an RL agent with keyboard control for UnitreeGo2 locomotion."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
from datetime import datetime

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL agent with keyboard control.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--enable_logging", action="store_true", default=True, help="Enable policy I/O logging.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import time
import torch
import numpy as np

import carb
import omni
from omni.kit.viewport.utility import get_viewport_from_window_name
from omni.kit.viewport.utility.camera_state import ViewportCameraState
from pxr import Gf, Sdf
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

# PLACEHOLDER: Extension template (do not remove this comment)


class PolicyLogger:
    """Policy I/O logger for play_go2_keyboard.py"""
    
    def __init__(self, log_dir: str = None, max_steps: int = 500):
        self.log_directory = log_dir
        self.log_file_path = None
        self.step_count = 0
        self.max_steps = max_steps
        self.flag = True
        
        if self.log_directory is None:
            self._auto_init_logging()
        else:
            self.set_log_directory(log_dir)
    
    def _auto_init_logging(self):
        """Automatically initialize logging directory and file"""
        try:
            # Create logs directory if it doesn't exist
            logs_base_dir = os.path.join(os.getcwd(), "logs")
            os.makedirs(logs_base_dir, exist_ok=True)
            
            # Create timestamped log directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_dir = os.path.join(logs_base_dir, f"play_policy_{timestamp}")
            os.makedirs(log_dir, exist_ok=True)
            
            # Set the log directory
            self.set_log_directory(log_dir)
            
        except Exception as e:
            print(f"❌ Error auto-initializing logging: {e}")
            self.log_file_path = None
    
    def set_log_directory(self, log_dir: str):
        """Set the log directory and create the policy_IO.py file"""
        self.log_directory = log_dir
        self.log_file_path = os.path.join(log_dir, "policy_IO.py")
        
        try:
            # Create the initial policy_IO.py file with proper structure
            with open(self.log_file_path, 'w') as f:
                f.write("# Policy Input/Output Log (from play_go2_keyboard.py)\n")
                f.write(f"# Created at: {datetime.now().isoformat()}\n\n")
                f.write("# observation & action configuration is based on sim:\n")
                f.write("# Each step contains:\n")
                f.write("# - step: step number\n")
                f.write("# - timestamp: ISO timestamp\n")
                f.write("# - policy input: observation (45 dims)\n")
                f.write("# - policy output: raw action (12 dims)\n")
                f.write("policy_steps = [\n")
            
            # Verify file was created
            if os.path.exists(self.log_file_path):
                print(f"✅ Policy logging initialized: {self.log_file_path}")
            else:
                print(f"❌ Failed to create log file: {self.log_file_path}")
                
        except Exception as e:
            print(f"❌ Error initializing policy logging: {e}")
            self.log_file_path = None
    
    def log_policy_step(self, obs: torch.Tensor, actions: torch.Tensor):
        """Log a single policy step to the policy_IO.py file"""
        if self.log_file_path is None:
            return
        
        # Check if we've reached the maximum number of steps
        if self.step_count >= self.max_steps:
            # Close the list with bracket when max steps reached
            if self.flag == True:
                self.finalize_log()
                self.flag = False
            return
            
        self.step_count += 1
        
        # Convert tensors to numpy arrays
        obs_np = obs.cpu().numpy()
        actions_np = actions.cpu().numpy()
        
        # Take first environment if multiple environments
        if obs_np.ndim > 1:
            obs_np = obs_np[0]
        if actions_np.ndim > 1:
            actions_np = actions_np[0]
        
        # Break down observation into components (Go2 specific)
        if len(obs_np) == 45:
            obs_components = {
                'base_ang_vel': obs_np[0:3].tolist(),
                'projected_gravity': obs_np[3:6].tolist(),
                'velocity_commands': obs_np[6:9].tolist(),
                'relative_joint_pos': obs_np[9:21].tolist(),
                'relative_joint_vel': obs_np[21:33].tolist(),
                'prev_actions': obs_np[33:45].tolist(),
                'concatenated_obs': obs_np.tolist()
            }
        else:
            # Fallback for non-Go2 observations
            obs_components = {
                'concatenated_obs': obs_np.tolist()
            }

        # Prepare step data
        step_data = {
            'step': self.step_count,
            'timestamp': datetime.now().isoformat(),
            'input': obs_components,
            'output': {
                'raw_actions': actions_np.flatten().tolist()
            }
        }
        
        # Append to file in Python format and flush immediately
        with open(self.log_file_path, 'a') as f:
            f.write(f"    # Step {self.step_count}\n")
            f.write(f"    {step_data},\n")
            f.flush()  # Ensure data is written to disk immediately
    
    def finalize_log(self):
        """Finalize the log file by closing the list"""
        if self.log_file_path and os.path.exists(self.log_file_path):
            try:
                with open(self.log_file_path, 'a') as f:
                    f.write("]\n\n")
                    f.write("# Log completed\n")
                    f.write(f"# Total steps: {self.step_count}\n")
                    f.write(f"# Completed at: {datetime.now().isoformat()}\n")
                
                print(f"✅ Policy log completed: {self.log_file_path} (Steps: {self.step_count})")
            except Exception as e:
                print(f"❌ Error finalizing log: {e}")


class Go2KeyboardDemo:
    """This class provides an interactive demo for UnitreeGo2 locomotion with keyboard control.
    It loads a pre-trained checkpoint and defines keyboard commands for directing motion.

    Keyboard controls:
    * UP: go forward
    * DOWN: go backward
    * LEFT: move left (strafe)
    * RIGHT: move right (strafe)
    * Z: turn left
    * X: turn right
    * SPACE: stop all motion
    * C: switch between third-person and perspective views
    * ESC: exit current third-person view
    """

    def __init__(self, task_name: str, args_cli):
        """Initialize the demo with environment and policy loading."""
        self.task_name = task_name
        self.args_cli = args_cli
        
        # Parse configuration
        env_cfg = parse_env_cfg(
            args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
        )
        agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(task_name, args_cli)

        # Modify environment configuration for interactive play
        env_cfg.episode_length_s = 1000000  # Very long episodes
        if hasattr(env_cfg, 'curriculum'):
            env_cfg.curriculum = None  # Disable curriculum
        
        # Create environment
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

        # Convert to single-agent instance if required
        if isinstance(env.unwrapped, DirectMARLEnv):
            env = multi_agent_to_single_agent(env)

        # Video recording setup
        if args_cli.video:
            video_kwargs = {
                "video_folder": "videos/go2_keyboard_play",
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording videos during play.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        # Wrap environment for rsl-rl
        self.env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        self.device = self.env.unwrapped.device

        # Load checkpoint
        if args_cli.use_pretrained_checkpoint:
            resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
            if not resume_path:
                print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
                return
        elif args_cli.checkpoint:
            resume_path = retrieve_file_path(args_cli.checkpoint)
        else:
            log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
            log_root_path = os.path.abspath(log_root_path)
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        
        # Load trained model
        ppo_runner = OnPolicyRunner(self.env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        ppo_runner.load(resume_path)
        self.policy = ppo_runner.get_inference_policy(device=self.device)

        # Export policy if needed
        try:
            policy_nn = ppo_runner.alg.policy
        except AttributeError:
            policy_nn = ppo_runner.alg.actor_critic

        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
        export_policy_as_onnx(
            policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx"
        )

        # Initialize policy logger
        self.policy_logger = None
        if args_cli.enable_logging:
            self.policy_logger = PolicyLogger(max_steps=500)

        # Initialize camera and keyboard setup
        self.create_camera()
        self.setup_keyboard()
        
        # Command buffer for velocity commands (vx, vy, omega_z)
        num_envs = self.env.unwrapped.num_envs
        self.commands = torch.zeros(num_envs, 3, device=self.device)
        
        # Selection and camera tracking
        self._prim_selection = omni.usd.get_context().get_selection()
        self._selected_id = None
        self._previous_selected_id = None
        self._camera_local_transform = torch.tensor([-1.5, 0.0, 0.5], device=self.device)  # Closer camera for smaller Go2

    def create_camera(self):
        """Creates a camera for third-person view."""
        stage = omni.usd.get_context().get_stage()
        self.viewport = get_viewport_from_window_name("Viewport")
        
        # Create camera
        self.camera_path = "/World/Camera"
        self.perspective_path = "/OmniverseKit_Persp"
        camera_prim = stage.DefinePrim(self.camera_path, "Camera")
        camera_prim.GetAttribute("focalLength").Set(8.5)
        coi_prop = camera_prim.GetProperty("omni:kit:centerOfInterest")
        if not coi_prop or not coi_prop.IsValid():
            camera_prim.CreateAttribute(
                "omni:kit:centerOfInterest", Sdf.ValueTypeNames.Vector3d, True, Sdf.VariabilityUniform
            ).Set(Gf.Vec3d(0, 0, -10))
        self.viewport.set_active_camera(self.perspective_path)

    def setup_keyboard(self):
        """Setup keyboard input handling."""
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub_keyboard = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)
        
        # Track which keys are currently pressed
        self._keys_pressed = set()
        
        # Define maximum values for each command type (adapted for Go2)
        self._max_values = {
            "linear_x": 0.5,   # Go2 max forward speed
            "linear_y": 0.5,   # Go2 max lateral speed  
            "angular_z": 0.5   # Go2 max angular speed
        }
        
        # Define acceleration rates (how fast commands ramp up per second)
        self._acceleration_rates = {
            "linear_x": 1.5,   # Reach max in 1.0 seconds
            "linear_y": 1.0,   # Reach max in 1.0 seconds  
            "angular_z": 1.0   # Reach max in 0.5 seconds
        }
        
        # Current target velocities based on key presses
        self._target_velocities = torch.zeros(3, device=self.device)  # [vx, vy, omega_z]
        
        # Smoothed current velocities (what actually gets sent)
        self._current_velocities = torch.zeros(3, device=self.device)
        
        # Define key mappings to velocity directions and magnitudes
        self._key_mappings = {
            "UP": (0, 1.0),      # linear_x positive (forward)
            "DOWN": (0, -1.0),   # linear_x negative (backward)
            "LEFT": (1, 1.0),    # linear_y positive (strafe left)
            "RIGHT": (1, -1.0),  # linear_y negative (strafe right)
            "Q": (2, 1.0),       # angular_z positive (turn left)
            "E": (2, -1.0),      # angular_z negative (turn right)
        }

    def _on_keyboard_event(self, event):
        """Handle keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            # Movement keys
            if event.input.name in self._key_mappings:
                self._keys_pressed.add(event.input.name)
            # Special keys
            elif event.input.name == "SPACE":
                # Reset all commands to zero
                self._keys_pressed.clear()
                self._target_velocities.fill_(0.0)
                self._current_velocities.fill_(0.0)
            elif event.input.name == "ESCAPE":
                self._prim_selection.clear_selected_prim_paths()
            elif event.input.name == "C":
                if self._selected_id is not None:
                    if self.viewport.get_active_camera() == self.camera_path:
                        self.viewport.set_active_camera(self.perspective_path)
                    else:
                        self.viewport.set_active_camera(self.camera_path)
        
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            # Remove key from pressed set
            if event.input.name in self._key_mappings:
                self._keys_pressed.discard(event.input.name)
    
    def update_velocities(self, dt):
        """Update velocity commands based on currently pressed keys."""
        # Reset target velocities
        self._target_velocities.fill_(0.0)
        
        # Calculate target velocities based on pressed keys
        for key in self._keys_pressed:
            if key in self._key_mappings:
                axis_idx, direction = self._key_mappings[key]
                axis_names = ["linear_x", "linear_y", "angular_z"]
                axis_name = axis_names[axis_idx]
                max_val = self._max_values[axis_name]
                self._target_velocities[axis_idx] += direction * max_val
        
        # Smoothly interpolate current velocities towards target velocities
        for i in range(3):
            axis_names = ["linear_x", "linear_y", "angular_z"]
            axis_name = axis_names[i]
            accel_rate = self._acceleration_rates[axis_name]
            
            # Calculate the maximum change this timestep
            max_change = accel_rate * dt
            
            # Calculate difference between target and current
            diff = self._target_velocities[i] - self._current_velocities[i]
            
            # Limit the change to max_change
            if abs(diff) <= max_change:
                self._current_velocities[i] = self._target_velocities[i]
            else:
                self._current_velocities[i] += torch.sign(diff) * max_change
        
        # Update command buffer for all environments
        if self._selected_id is not None:
            # Only update selected robot
            self.commands[self._selected_id] = self._current_velocities.clone()
        else:
            # Update all robots
            self.commands[:] = self._current_velocities.clone()

    def update_selected_object(self):
        """Update selected robot for camera tracking."""
        self._previous_selected_id = self._selected_id
        selected_prim_paths = self._prim_selection.get_selected_prim_paths()
        
        if len(selected_prim_paths) == 0:
            self._selected_id = None
            self.viewport.set_active_camera(self.perspective_path)
        elif len(selected_prim_paths) > 1:
            print("Multiple prims selected. Please select only one!")
        else:
            prim_splitted_path = selected_prim_paths[0].split("/")
            # Check if a valid robot was selected
            if len(prim_splitted_path) >= 4 and prim_splitted_path[3].startswith("env_"):
                self._selected_id = int(prim_splitted_path[3][4:])
                if self._previous_selected_id != self._selected_id:
                    self.viewport.set_active_camera(self.camera_path)
                self._update_camera()
            else:
                print("The selected prim was not a valid robot")

    def _update_camera(self):
        """Update camera to follow selected robot."""
        if self._selected_id is None:
            return
            
        # Get robot base position and orientation
        robot = self.env.unwrapped.scene["robot"]
        base_pos = robot.data.root_pos_w[self._selected_id, :]
        base_quat = robot.data.root_quat_w[self._selected_id, :]

        # Calculate camera position
        camera_pos = quat_apply(base_quat, self._camera_local_transform) + base_pos

        # Update camera state
        camera_state = ViewportCameraState(self.camera_path, self.viewport)
        eye = Gf.Vec3d(camera_pos[0].item(), camera_pos[1].item(), camera_pos[2].item())
        target = Gf.Vec3d(base_pos[0].item(), base_pos[1].item(), base_pos[2].item() + 0.3)  # Lower target for Go2
        camera_state.set_position_world(eye, True)
        camera_state.set_target_world(target, True)

    def run(self):
        """Main execution loop."""
        print("\n" + "="*50)
        print("UnitreeGo2 Keyboard Control Demo")
        print("="*50)
        print(f"Environment: {args_cli.task}")
        print("Keyboard Controls:")
        print("  UP Arrow    : Move forward (max speed: 2.0 m/s)")
        print("  DOWN Arrow  : Move backward (max speed: -2.0 m/s)")
        print("  LEFT Arrow  : Strafe left (max speed: 1.0 m/s)")
        print("  RIGHT Arrow : Strafe right (max speed: -1.0 m/s)")
        print("  Q           : Turn left (max angular: 1.5 rad/s)")
        print("  E           : Turn right (max angular: -1.5 rad/s)")
        print("  SPACE       : Reset all commands to zero")
        print("  C           : Switch camera view (when robot selected)")
        print("  ESC         : Clear selection")
        print("\nNote: Hold keys to gradually increase speed. Release to gradually decrease.")
        print("\nClick on a robot to select it for individual control.")
        print("Without selection, all robots are controlled together.")
        print("\nSupported environments with custom observations (45/48-dim):")
        print("  - Isaac-Velocity-Sequor-RNN-Unitree-Go2-Play-v0")
        print("  - Isaac-Velocity-Sequor-Unitree-Go2-Play-v0")
        print("  - Isaac-Velocity-Sequor-Rough-RNN-Unitree-Go2-Play-v0")
        print("  - Isaac-Velocity-Sequor-Rough-Unitree-Go2-Play-v0")
        print("="*50 + "\n")

        dt = self.env.unwrapped.step_dt
        obs, _ = self.env.get_observations()
        timestep = 0

        # Simulation loop
        while simulation_app.is_running():
            start_time = time.time()
            
            # Update selected robot
            self.update_selected_object()
            
            # Update velocities based on keyboard input
            self.update_velocities(dt)
            
            # Update command manager before getting observations
            try:
                # Set commands using the command manager
                if hasattr(self.env.unwrapped, 'command_manager'):
                    # For environments with command manager, set specific robot commands
                    if self._selected_id is not None:
                        # Create command tensor for specific robot
                        cmd_tensor = torch.zeros(self.env.unwrapped.num_envs, 3, device=self.device)
                        cmd_tensor[self._selected_id] = self.commands[self._selected_id]
                        self.env.unwrapped.command_manager.set_command("base_velocity", cmd_tensor)
                    else:
                        # Set commands for all robots
                        self.env.unwrapped.command_manager.set_command("base_velocity", self.commands)
            except Exception as e:
                print(f"Warning: Could not set commands via command manager: {e}")
            
            with torch.inference_mode():
                # Get policy action
                actions = self.policy(obs)
                
                # Log policy step if logging is enabled
                if self.policy_logger is not None:
                    self.policy_logger.log_policy_step(obs, actions)
                
                # Step environment
                obs, _, _, _ = self.env.step(actions)
                
                # As a backup, also override the command portion of observation
                # For custom observations: velocity commands are at indices 6:9 in 45-dim Go2 observation
                try:
                    # Check observation dimensions and set commands accordingly
                    if obs.shape[-1] == 45:  # Custom observation (policy)
                        obs[:, 6:9] = self.commands
                    elif obs.shape[-1] == 48:  # Custom observation (critic) - shouldn't happen in policy inference
                        obs[:, 6:9] = self.commands
                    else:
                        print(f"Warning: Unexpected observation dimension {obs.shape[-1]}")
                        print(f"Expected 45 (policy) for custom Go2 environments")
                        print(f"Current commands will be applied via command manager only")
                except Exception as e:
                    print(f"Warning: Could not override commands in observation: {e}")
                    pass

            if args_cli.video:
                timestep += 1
                if timestep == args_cli.video_length:
                    break

            # Real-time control
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)

        # Finalize logging
        if self.policy_logger is not None:
            self.policy_logger.finalize_log()

        # Cleanup
        self.env.close()


def main():
    """Main function."""
    task_name = args_cli.task.split(":")[-1]
    demo = Go2KeyboardDemo(task_name, args_cli)
    demo.run()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close() 