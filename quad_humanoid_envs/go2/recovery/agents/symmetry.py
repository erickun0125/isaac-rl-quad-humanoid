"""Symmetry functions for GO2 recovery locomotion training.

Go2 Joint Order:
[FL_hip, FR_hip, RL_hip, RR_hip,
 FL_thigh, FR_thigh, RL_thigh, RR_thigh,
 FL_calf, FR_calf, RL_calf, RR_calf]

Policy obs structure (84 dims):
- base_ang_vel: 6 (0:6) - 3dims x 2history
- projected_gravity: 6 (6:12) - 3dims x 2history
- joint_pos: 24 (12:36) - 12joints x 2history
- joint_vel: 24 (36:60) - 12joints x 2history
- actions: 24 (60:84) - 12joints x 2history

Critic obs structure (85 dims):
- base_ang_vel: 6 (0:6) - 3dims x 2history
- projected_gravity: 6 (6:12) - 3dims x 2history
- joint_pos: 24 (12:36) - 12joints x 2history
- joint_vel: 24 (36:60) - 12joints x 2history
- actions: 24 (60:84) - 12joints x 2history
- progress_ratio: 1 (84:85) - normalized progress (0-1)
"""

import torch


def custom_locomotion_symmetry(env, obs, actions, obs_type="policy"):
    """Data augmentation via left-right symmetry for Go2 recovery observations."""
    if obs is not None:
        batch_size = obs.shape[0]
        obs_mirrored = torch.cat([obs, obs.clone()], dim=0)
        obs_mirrored[batch_size:] = mirror_observations(obs_mirrored[batch_size:], obs_type)
    else:
        obs_mirrored = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_mirrored = torch.cat([actions, actions.clone()], dim=0)
        actions_mirrored[batch_size:] = mirror_actions(actions_mirrored[batch_size:])
    else:
        actions_mirrored = None

    return obs_mirrored, actions_mirrored


def mirror_observations(obs, obs_type):
    """Mirror observations using left-right symmetry."""
    mirrored = obs.clone()

    if obs_type == "policy":
        # base_ang_vel with history (0:6): 6 = 3dims x 2history
        mirror_base_ang_vel_history(mirrored, start_idx=0, history_len=2)

        # projected_gravity with history (6:12): 6 = 3dims x 2history
        mirror_projected_gravity_history(mirrored, start_idx=6, history_len=2)

        # joint_pos with history (12:36): 24 = 12joints x 2history
        mirror_joint_history(mirrored, start_idx=12, joint_dim=12, history_len=2)

        # joint_vel with history (36:60): 24 = 12joints x 2history
        mirror_joint_history(mirrored, start_idx=36, joint_dim=12, history_len=2)

        # actions with history (60:84): 24 = 12joints x 2history
        mirror_joint_history(mirrored, start_idx=60, joint_dim=12, history_len=2)

    elif obs_type == "critic":
        # base_ang_vel with history (0:6)
        mirror_base_ang_vel_history(mirrored, start_idx=0, history_len=2)

        # projected_gravity with history (6:12)
        mirror_projected_gravity_history(mirrored, start_idx=6, history_len=2)

        # joint_pos with history (12:36)
        mirror_joint_history(mirrored, start_idx=12, joint_dim=12, history_len=2)

        # joint_vel with history (36:60)
        mirror_joint_history(mirrored, start_idx=36, joint_dim=12, history_len=2)

        # actions with history (60:84)
        mirror_joint_history(mirrored, start_idx=60, joint_dim=12, history_len=2)

        # progress_ratio (84:85): scalar, no mirroring needed

    return mirrored


def mirror_joint_history(tensor, start_idx, joint_dim, history_len):
    """Mirror flattened joint data with history for Go2 (FL<->FR, RL<->RR, hip sign flip)."""
    for h in range(history_len):
        time_start = start_idx + h * joint_dim
        joint_data = tensor[:, time_start:time_start + joint_dim].clone()

        # Hip joints (0-3): FL<->FR, RL<->RR
        tensor[:, time_start + 0] = joint_data[:, 1]
        tensor[:, time_start + 1] = joint_data[:, 0]
        tensor[:, time_start + 2] = joint_data[:, 3]
        tensor[:, time_start + 3] = joint_data[:, 2]

        # Thigh joints (4-7): FL<->FR, RL<->RR
        tensor[:, time_start + 4] = joint_data[:, 5]
        tensor[:, time_start + 5] = joint_data[:, 4]
        tensor[:, time_start + 6] = joint_data[:, 7]
        tensor[:, time_start + 7] = joint_data[:, 6]

        # Calf joints (8-11): FL<->FR, RL<->RR
        tensor[:, time_start + 8] = joint_data[:, 9]
        tensor[:, time_start + 9] = joint_data[:, 8]
        tensor[:, time_start + 10] = joint_data[:, 11]
        tensor[:, time_start + 11] = joint_data[:, 10]

        # Hip sign flip (lateral rotation)
        tensor[:, time_start + 0] *= -1
        tensor[:, time_start + 1] *= -1
        tensor[:, time_start + 2] *= -1
        tensor[:, time_start + 3] *= -1


def mirror_actions(actions):
    """Mirror actions using left-right symmetry for Go2."""
    mirrored = actions.clone()
    action_temp = mirrored.clone()

    # Hip joints (0-3): FL<->FR, RL<->RR
    mirrored[:, 0] = action_temp[:, 1]
    mirrored[:, 1] = action_temp[:, 0]
    mirrored[:, 2] = action_temp[:, 3]
    mirrored[:, 3] = action_temp[:, 2]

    # Thigh joints (4-7): FL<->FR, RL<->RR
    mirrored[:, 4] = action_temp[:, 5]
    mirrored[:, 5] = action_temp[:, 4]
    mirrored[:, 6] = action_temp[:, 7]
    mirrored[:, 7] = action_temp[:, 6]

    # Calf joints (8-11): FL<->FR, RL<->RR
    mirrored[:, 8] = action_temp[:, 9]
    mirrored[:, 9] = action_temp[:, 8]
    mirrored[:, 10] = action_temp[:, 11]
    mirrored[:, 11] = action_temp[:, 10]

    # Hip sign flip (lateral rotation)
    mirrored[:, 0] *= -1
    mirrored[:, 1] *= -1
    mirrored[:, 2] *= -1
    mirrored[:, 3] *= -1

    return mirrored


def mirror_base_ang_vel_history(tensor, start_idx, history_len):
    """Mirror flattened base angular velocity: (x, y, z) -> (-x, y, -z) per history step."""
    for h in range(history_len):
        time_start = start_idx + h * 3
        tensor[:, time_start + 0] *= -1  # ang_vel_x
        tensor[:, time_start + 2] *= -1  # ang_vel_z


def mirror_projected_gravity_history(tensor, start_idx, history_len):
    """Mirror flattened projected gravity: (x, y, z) -> (x, -y, z) per history step."""
    for h in range(history_len):
        time_start = start_idx + h * 3
        tensor[:, time_start + 1] *= -1  # gravity_y
