"""Left-right symmetry functions for G1 loco-manipulation with end-effector pose commands.

Policy obs structure (WITH EE POSE COMMANDS):
- base_ang_vel: 3 dims (0:3)
- projected_gravity: 3 dims (3:6)
- velocity_commands: 3 dims (6:9)
- left_ee_pose_command: 7 dims (9:16) - [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
- right_ee_pose_command: 7 dims (16:23) - [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
- joint_pos: 87 dims (23:110) - history=3, flatten (29 joints * 3)
- joint_vel: 58 dims (110:168) - history=2, flatten (29 joints * 2)
- actions: 29 dims (168:197) - history=1, flatten (29 joints * 1)

Critic obs structure (WITH EE POSE COMMANDS):
- base_lin_vel: 3 dims (0:3)
- base_pos_z: 1 dim (3:4)
- foot_contact: 2 dims (4:6) - left, right foot contact
- base_ang_vel: 3 dims (6:9)
- projected_gravity: 3 dims (9:12)
- velocity_commands: 3 dims (12:15)
- left_ee_pose_command: 7 dims (15:22) - [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
- right_ee_pose_command: 7 dims (22:29) - [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]
- joint_pos: 87 dims (29:116) - history=3, flatten (29 joints * 3)
- joint_vel: 58 dims (116:174) - history=2, flatten (29 joints * 2)
- actions: 29 dims (174:203) - history=1, flatten (29 joints * 1)
"""

import torch
from tensordict import TensorDict

from .symmetry import mirror_actions, mirror_joint_history


def g1_locomanip_symmetry_with_ee(env, obs, actions, obs_type="policy"):  # noqa: ARG001
    """G1 humanoid robot symmetry function with end-effector pose commands.

    Doubles the batch by appending mirrored copies of observations and actions.
    Supports both regular tensors and TensorDict observations.
    """
    if obs is not None:
        if isinstance(obs, TensorDict):
            batch_size = obs.batch_size[0]
            obs_aug = obs.repeat(2)

            obs_aug["policy"][:batch_size] = obs["policy"][:]
            obs_aug["critic"][:batch_size] = obs["critic"][:]

            obs_aug["policy"][batch_size:] = mirror_observations_with_ee(obs["policy"][:], "policy")
            obs_aug["critic"][batch_size:] = mirror_observations_with_ee(obs["critic"][:], "critic")

            obs_mirrored = obs_aug
        else:
            obs_batch_size = obs.shape[0]
            obs_mirrored = torch.cat([obs, obs.clone()], dim=0)
            obs_mirrored[obs_batch_size:] = mirror_observations_with_ee(obs_mirrored[obs_batch_size:], obs_type)
    else:
        obs_mirrored = None

    if actions is not None:
        action_batch_size = actions.shape[0]
        actions_mirrored = torch.cat([actions, actions.clone()], dim=0)
        actions_mirrored[action_batch_size:] = mirror_actions(actions_mirrored[action_batch_size:])
    else:
        actions_mirrored = None

    return obs_mirrored, actions_mirrored


def mirror_observations_with_ee(obs, obs_type):
    """Mirror observations including end-effector pose commands."""
    mirrored = obs.clone()

    if obs_type == "policy":
        # base_ang_vel (0:3): (x, y, z) -> (-x, y, -z)
        mirrored[:, 0] *= -1  # ang_vel_x
        mirrored[:, 2] *= -1  # ang_vel_z

        # projected_gravity (3:6): (x, y, z) -> (x, -y, z)
        mirrored[:, 4] *= -1  # gravity_y

        # velocity_commands (6:9): (vx, vy, vyaw) -> (vx, -vy, -vyaw)
        mirrored[:, 7] *= -1   # vy
        mirrored[:, 8] *= -1   # vyaw

        # left_ee_pose_command (9:16) <-> right_ee_pose_command (16:23)
        mirror_ee_pose_commands(mirrored, left_start=9, right_start=16)

        # joint_pos with history (23:110): 87 dims = 29 joints * 3 history
        mirror_joint_history(mirrored, start_idx=23, joint_dim=29, history_len=3)

        # joint_vel with history (110:168): 58 dims = 29 joints * 2 history
        mirror_joint_history(mirrored, start_idx=110, joint_dim=29, history_len=2)

        # actions with history (168:197): 29 dims = 29 joints * 1 history
        mirror_joint_history(mirrored, start_idx=168, joint_dim=29, history_len=1)

    elif obs_type == "critic":
        # base_lin_vel (0:3): (x, y, z) -> (x, -y, z)
        mirrored[:, 1] *= -1  # lin_vel_y

        # base_pos_z (3:4): no change

        # foot_contact (4:6): (left, right) -> (right, left)
        temp_left_contact = mirrored[:, 4].clone()
        mirrored[:, 4] = mirrored[:, 5]
        mirrored[:, 5] = temp_left_contact

        # base_ang_vel (6:9): (x, y, z) -> (-x, y, -z)
        mirrored[:, 6] *= -1  # ang_vel_x
        mirrored[:, 8] *= -1  # ang_vel_z

        # projected_gravity (9:12): (x, y, z) -> (x, -y, z)
        mirrored[:, 10] *= -1  # gravity_y

        # velocity_commands (12:15): (vx, vy, vyaw) -> (vx, -vy, -vyaw)
        mirrored[:, 13] *= -1  # vy
        mirrored[:, 14] *= -1  # vyaw

        # left_ee_pose_command (15:22) <-> right_ee_pose_command (22:29)
        mirror_ee_pose_commands(mirrored, left_start=15, right_start=22)

        # joint_pos with history (29:116): 87 dims = 29 joints * 3 history
        mirror_joint_history(mirrored, start_idx=29, joint_dim=29, history_len=3)

        # joint_vel with history (116:174): 58 dims = 29 joints * 2 history
        mirror_joint_history(mirrored, start_idx=116, joint_dim=29, history_len=2)

        # actions with history (174:203): 29 dims = 29 joints * 1 history
        mirror_joint_history(mirrored, start_idx=174, joint_dim=29, history_len=1)

    return mirrored


def mirror_ee_pose_commands(tensor, left_start, right_start):
    """Mirror end-effector pose commands for left-right symmetry.

    Pose format: [pos_x, pos_y, pos_z, quat_w, quat_x, quat_y, quat_z]

    Position mirroring: (x, y, z) -> (x, -y, z)
    Quaternion mirroring for y-axis reflection: (w, x, y, z) -> (w, -x, y, -z)
    """
    left_pose = tensor[:, left_start:left_start+7].clone()
    right_pose = tensor[:, right_start:right_start+7].clone()

    # Swap left and right poses
    tensor[:, left_start:left_start+7] = right_pose
    tensor[:, right_start:right_start+7] = left_pose

    # Mirror left pose (now contains original right pose)
    tensor[:, left_start+1] *= -1  # pos_y
    tensor[:, left_start+4] *= -1  # quat_x
    tensor[:, left_start+6] *= -1  # quat_z

    # Mirror right pose (now contains original left pose)
    tensor[:, right_start+1] *= -1  # pos_y
    tensor[:, right_start+4] *= -1  # quat_x
    tensor[:, right_start+6] *= -1  # quat_z
