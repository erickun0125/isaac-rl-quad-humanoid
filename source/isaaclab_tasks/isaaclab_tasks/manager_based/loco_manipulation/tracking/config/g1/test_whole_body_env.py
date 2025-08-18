#!/usr/bin/env python3

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Test script for G1 Whole Body Control Environment.

This script demonstrates how to use the whole body control environment where:
- 4 joint groups: Hand, Arm, Waist, Leg
- 3 policy types per group: RL, IL, IK
- Configurable policy assignment per group

Usage:
    python test_whole_body_env.py --env Isaac-Tracking-WholeBody-G1-LowerBodyRL-v0 --num_envs 64
"""

import argparse
import torch

import gymnasium as gym

import isaaclab.envs  # noqa: F401
import isaaclab_tasks  # noqa: F401


def main():
    """Run the test for whole body control environment."""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test G1 Whole Body Control Environment")
    parser.add_argument(
        "--env", 
        type=str, 
        default="Isaac-Tracking-WholeBody-G1-LowerBodyRL-v0",
        choices=[
            "Isaac-Tracking-WholeBody-G1-v0",
            "Isaac-Tracking-WholeBody-G1-Play-v0", 
            "Isaac-Tracking-WholeBody-G1-LowerBodyRL-v0",
            "Isaac-Tracking-WholeBody-G1-UpperBodyIL-v0",
            "Isaac-Tracking-WholeBody-G1-FullRL-v0",
        ],
        help="Environment to test"
    )
    parser.add_argument("--num_envs", type=int, default=64, help="Number of environments to create")
    parser.add_argument("--enable_cameras", action="store_true", help="Enable camera sensors")
    parser.add_argument("--episode_length", type=int, default=1000, help="Episode length")
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print(f"Testing G1 Whole Body Control Environment: {args.env}")
    print("=" * 80)
    
    # Create environment
    print(f"\nCreating environment with {args.num_envs} environments...")
    env = gym.make(args.env, num_envs=args.num_envs, enable_cameras=args.enable_cameras)
    
    # Print environment information
    print(f"\nEnvironment: {env.spec.id}")
    print(f"Number of environments: {env.num_envs}")
    print(f"Episode length: {env.max_episode_length}")
    print(f"Action space dimension: {env.action_space}")
    print(f"Observation space dimensions:")
    for group, space in env.observation_space.items():
        print(f"  - {group}: {space}")
    
    # Print policy configuration information
    action_manager = env.action_manager
    if hasattr(action_manager._terms["joint_pos"], "get_group_policy"):  # pylint: disable=protected-access
        print("\nPolicy Configuration:")
        from isaaclab_tasks.manager_based.loco_manipulation.tracking.config.g1.mdp.whole_body_actions import JointGroup
        for group in [JointGroup.HAND, JointGroup.ARM, JointGroup.WAIST, JointGroup.LEG]:
            policy = action_manager._terms["joint_pos"].get_group_policy(group)  # pylint: disable=protected-access
            joint_names = action_manager._terms["joint_pos"].get_group_joint_names(group)  # pylint: disable=protected-access
            print(f"  - {group.value.upper()}: {policy.value.upper()} ({len(joint_names)} joints)")
        
        print(f"\nRL-controlled action dimension: {action_manager._terms['joint_pos'].action_dim}")  # pylint: disable=protected-access
    
    # Reset environment
    print("\nResetting environment...")
    obs, _ = env.reset()
    print("Initial observation shapes:")
    for group, data in obs.items():
        print(f"  - {group}: {data.shape}")
    
    # Simulation loop
    print(f"\nRunning simulation for {args.episode_length} steps...")
    for step in range(args.episode_length):
        # Generate random actions for RL-controlled joints only
        # The action space dimension depends on which groups use RL policy
        actions = torch.rand(env.action_space.shape, device=env.device) * 0.2 - 0.1  # Small random actions
        
        # Step environment
        obs, rewards, terminated, truncated, _ = env.step(actions)
        
        # Print progress
        if step % 100 == 0:
            print(f"Step {step:4d}: Mean reward = {rewards.mean():.3f}, "
                  f"Terminated = {terminated.sum().item()}, "
                  f"Truncated = {truncated.sum().item()}")
        
        # Reset terminated environments
        if terminated.any() or truncated.any():
            env.reset()
    
    print("\nSimulation completed successfully!")
    print("=" * 80)
    
    # Close environment
    env.close()


if __name__ == "__main__":
    main()
