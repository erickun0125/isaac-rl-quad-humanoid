"""G1 humanoid robot symmetry functions for whole body control.

Observation structure (Policy):
- base_ang_vel: 3 dims (0:3)
- projected_gravity: 3 dims (3:6)
- velocity_commands: 3 dims (6:9)
- joint_pos: 87 dims (9:96) - history=3, flatten (29 joints * 3)
- joint_vel: 58 dims (96:154) - history=2, flatten (29 joints * 2)
- actions: 29 dims (154:183) - history=1, flatten (29 joints * 1)

Observation structure (Critic):
- base_lin_vel: 3 dims (0:3)
- base_pos_z: 1 dim (3:4)
- base_ang_vel: 3 dims (4:7)
- projected_gravity: 3 dims (7:10)
- velocity_commands: 3 dims (10:13)
- joint_pos: 87 dims (13:100) - history=3, flatten (29 joints * 3)
- joint_vel: 58 dims (100:158) - history=2, flatten (29 joints * 2)
- actions: 29 dims (158:187) - history=1, flatten (29 joints * 1)

G1 Joint Order (29 DOF) - ACTUAL VERIFIED FROM ENVIRONMENT:
0: waist_pitch_joint
1: left_shoulder_pitch_joint <-> 2: right_shoulder_pitch_joint
3: waist_roll_joint
4: left_shoulder_roll_joint <-> 5: right_shoulder_roll_joint
6: waist_yaw_joint
7: left_shoulder_yaw_joint <-> 8: right_shoulder_yaw_joint
9: left_hip_pitch_joint <-> 10: right_hip_pitch_joint
11: left_elbow_joint <-> 12: right_elbow_joint
13: left_hip_roll_joint <-> 14: right_hip_roll_joint
15: left_wrist_roll_joint <-> 16: right_wrist_roll_joint
17: left_hip_yaw_joint <-> 18: right_hip_yaw_joint
19: left_wrist_pitch_joint <-> 20: right_wrist_pitch_joint
21: left_knee_joint <-> 22: right_knee_joint
23: left_wrist_yaw_joint <-> 24: right_wrist_yaw_joint
25: left_ankle_pitch_joint <-> 26: right_ankle_pitch_joint
27: left_ankle_roll_joint <-> 28: right_ankle_roll_joint
"""

import torch


def g1_wholebody_symmetry(env, obs, actions, obs_type="policy"):  # noqa: ARG001
    """G1 humanoid robot symmetry function for whole body control environment.

    Doubles the batch by appending a left-right mirrored copy.

    Args:
        env: Environment instance (unused).
        obs: Observation tensor.
        actions: Action tensor.
        obs_type: "policy" or "critic".

    Returns:
        Tuple of (mirrored_obs, mirrored_actions) with doubled batch size.
    """
    if obs is not None:
        obs_batch_size = obs.shape[0]
        obs_mirrored = torch.cat([obs, obs.clone()], dim=0)
        obs_mirrored[obs_batch_size:] = mirror_observations(obs_mirrored[obs_batch_size:], obs_type)
    else:
        obs_mirrored = None

    if actions is not None:
        action_batch_size = actions.shape[0]
        actions_mirrored = torch.cat([actions, actions.clone()], dim=0)
        actions_mirrored[action_batch_size:] = mirror_actions(actions_mirrored[action_batch_size:])
    else:
        actions_mirrored = None

    return obs_mirrored, actions_mirrored


def mirror_observations(obs, obs_type):
    """Mirror observations for left-right symmetry transformation."""
    mirrored = obs.clone()

    if obs_type == "policy":
        # base_ang_vel (0:3): (x, y, z) -> (-x, y, -z)
        mirrored[:, 0] *= -1
        mirrored[:, 2] *= -1

        # projected_gravity (3:6): (x, y, z) -> (x, -y, z)
        mirrored[:, 4] *= -1

        # velocity_commands (6:9): (vx, vy, vyaw) -> (vx, -vy, -vyaw)
        mirrored[:, 7] *= -1
        mirrored[:, 8] *= -1

        # joint_pos with history (9:96): 87 dims = 29 joints * 3 history
        mirror_joint_history(mirrored, start_idx=9, joint_dim=29, history_len=3)

        # joint_vel with history (96:154): 58 dims = 29 joints * 2 history
        mirror_joint_history(mirrored, start_idx=96, joint_dim=29, history_len=2)

        # actions with history (154:183): 29 dims = 29 joints * 1 history
        mirror_joint_history(mirrored, start_idx=154, joint_dim=29, history_len=1)

    elif obs_type == "critic":
        # base_lin_vel (0:3): (x, y, z) -> (x, -y, z)
        mirrored[:, 1] *= -1

        # base_pos_z (3:4): no change

        # base_ang_vel (4:7): (x, y, z) -> (-x, y, -z)
        mirrored[:, 4] *= -1
        mirrored[:, 6] *= -1

        # projected_gravity (7:10): (x, y, z) -> (x, -y, z)
        mirrored[:, 8] *= -1

        # velocity_commands (10:13): (vx, vy, vyaw) -> (vx, -vy, -vyaw)
        mirrored[:, 11] *= -1
        mirrored[:, 12] *= -1

        # joint_pos with history (13:100): 87 dims = 29 joints * 3 history
        mirror_joint_history(mirrored, start_idx=13, joint_dim=29, history_len=3)

        # joint_vel with history (100:158): 58 dims = 29 joints * 2 history
        mirror_joint_history(mirrored, start_idx=100, joint_dim=29, history_len=2)

        # actions with history (158:187): 29 dims = 29 joints * 1 history
        mirror_joint_history(mirrored, start_idx=158, joint_dim=29, history_len=1)

    return mirrored


def mirror_joint_history(tensor, start_idx, joint_dim, history_len):
    """Mirror joint data with history for G1 humanoid robot (29 DOF).

    Args:
        tensor: Observation tensor.
        start_idx: Joint data start index.
        joint_dim: Joint dimension (29).
        history_len: History length (1, 2, or 3).
    """
    for h in range(history_len):
        time_start = start_idx + h * joint_dim

        joint_data = tensor[:, time_start:time_start + joint_dim].clone()

        # Waist joints (central, sign flip only)
        # 0: waist_pitch_joint - no change
        # 3: waist_roll_joint - sign flip
        # 6: waist_yaw_joint - sign flip
        tensor[:, time_start + 3] *= -1
        tensor[:, time_start + 6] *= -1

        # Shoulder pitch: 1 <-> 2
        tensor[:, time_start + 1] = joint_data[:, 2]
        tensor[:, time_start + 2] = joint_data[:, 1]

        # Shoulder roll: 4 <-> 5 (sign flip)
        tensor[:, time_start + 4] = joint_data[:, 5]
        tensor[:, time_start + 5] = joint_data[:, 4]
        tensor[:, time_start + 4] *= -1
        tensor[:, time_start + 5] *= -1

        # Shoulder yaw: 7 <-> 8 (sign flip)
        tensor[:, time_start + 7] = joint_data[:, 8]
        tensor[:, time_start + 8] = joint_data[:, 7]
        tensor[:, time_start + 7] *= -1
        tensor[:, time_start + 8] *= -1

        # Hip pitch: 9 <-> 10
        tensor[:, time_start + 9] = joint_data[:, 10]
        tensor[:, time_start + 10] = joint_data[:, 9]

        # Elbow: 11 <-> 12
        tensor[:, time_start + 11] = joint_data[:, 12]
        tensor[:, time_start + 12] = joint_data[:, 11]

        # Hip roll: 13 <-> 14 (sign flip)
        tensor[:, time_start + 13] = joint_data[:, 14]
        tensor[:, time_start + 14] = joint_data[:, 13]
        tensor[:, time_start + 13] *= -1
        tensor[:, time_start + 14] *= -1

        # Wrist roll: 15 <-> 16 (sign flip)
        tensor[:, time_start + 15] = joint_data[:, 16]
        tensor[:, time_start + 16] = joint_data[:, 15]
        tensor[:, time_start + 15] *= -1
        tensor[:, time_start + 16] *= -1

        # Hip yaw: 17 <-> 18 (sign flip)
        tensor[:, time_start + 17] = joint_data[:, 18]
        tensor[:, time_start + 18] = joint_data[:, 17]
        tensor[:, time_start + 17] *= -1
        tensor[:, time_start + 18] *= -1

        # Wrist pitch: 19 <-> 20
        tensor[:, time_start + 19] = joint_data[:, 20]
        tensor[:, time_start + 20] = joint_data[:, 19]

        # Knee: 21 <-> 22
        tensor[:, time_start + 21] = joint_data[:, 22]
        tensor[:, time_start + 22] = joint_data[:, 21]

        # Wrist yaw: 23 <-> 24 (sign flip)
        tensor[:, time_start + 23] = joint_data[:, 24]
        tensor[:, time_start + 24] = joint_data[:, 23]
        tensor[:, time_start + 23] *= -1
        tensor[:, time_start + 24] *= -1

        # Ankle pitch: 25 <-> 26
        tensor[:, time_start + 25] = joint_data[:, 26]
        tensor[:, time_start + 26] = joint_data[:, 25]

        # Ankle roll: 27 <-> 28 (sign flip)
        tensor[:, time_start + 27] = joint_data[:, 28]
        tensor[:, time_start + 28] = joint_data[:, 27]
        tensor[:, time_start + 27] *= -1
        tensor[:, time_start + 28] *= -1


def mirror_actions(actions):
    """Mirror actions for left-right symmetry transformation (G1 humanoid robot)."""
    mirrored = actions.clone()
    action_temp = mirrored.clone()

    # Waist joints (central, sign flip only)
    mirrored[:, 3] *= -1  # waist_roll_joint
    mirrored[:, 6] *= -1  # waist_yaw_joint

    # Shoulder pitch: 1 <-> 2
    mirrored[:, 1] = action_temp[:, 2]
    mirrored[:, 2] = action_temp[:, 1]

    # Shoulder roll: 4 <-> 5 (sign flip)
    mirrored[:, 4] = action_temp[:, 5]
    mirrored[:, 5] = action_temp[:, 4]
    mirrored[:, 4] *= -1
    mirrored[:, 5] *= -1

    # Shoulder yaw: 7 <-> 8 (sign flip)
    mirrored[:, 7] = action_temp[:, 8]
    mirrored[:, 8] = action_temp[:, 7]
    mirrored[:, 7] *= -1
    mirrored[:, 8] *= -1

    # Hip pitch: 9 <-> 10
    mirrored[:, 9] = action_temp[:, 10]
    mirrored[:, 10] = action_temp[:, 9]

    # Elbow: 11 <-> 12
    mirrored[:, 11] = action_temp[:, 12]
    mirrored[:, 12] = action_temp[:, 11]

    # Hip roll: 13 <-> 14 (sign flip)
    mirrored[:, 13] = action_temp[:, 14]
    mirrored[:, 14] = action_temp[:, 13]
    mirrored[:, 13] *= -1
    mirrored[:, 14] *= -1

    # Wrist roll: 15 <-> 16 (sign flip)
    mirrored[:, 15] = action_temp[:, 16]
    mirrored[:, 16] = action_temp[:, 15]
    mirrored[:, 15] *= -1
    mirrored[:, 16] *= -1

    # Hip yaw: 17 <-> 18 (sign flip)
    mirrored[:, 17] = action_temp[:, 18]
    mirrored[:, 18] = action_temp[:, 17]
    mirrored[:, 17] *= -1
    mirrored[:, 18] *= -1

    # Wrist pitch: 19 <-> 20
    mirrored[:, 19] = action_temp[:, 20]
    mirrored[:, 20] = action_temp[:, 19]

    # Knee: 21 <-> 22
    mirrored[:, 21] = action_temp[:, 22]
    mirrored[:, 22] = action_temp[:, 21]

    # Wrist yaw: 23 <-> 24 (sign flip)
    mirrored[:, 23] = action_temp[:, 24]
    mirrored[:, 24] = action_temp[:, 23]
    mirrored[:, 23] *= -1
    mirrored[:, 24] *= -1

    # Ankle pitch: 25 <-> 26
    mirrored[:, 25] = action_temp[:, 26]
    mirrored[:, 26] = action_temp[:, 25]

    # Ankle roll: 27 <-> 28 (sign flip)
    mirrored[:, 27] = action_temp[:, 28]
    mirrored[:, 28] = action_temp[:, 27]
    mirrored[:, 27] *= -1
    mirrored[:, 28] *= -1

    return mirrored
