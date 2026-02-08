"""Left-right symmetry functions for G1 humanoid loco-manipulation environment.

Policy obs structure:
- base_ang_vel: 3 dims (0:3)
- projected_gravity: 3 dims (3:6)
- velocity_commands: 3 dims (6:9)
- joint_pos: 87 dims (9:96) - history=3, flatten (29 joints * 3)
- joint_vel: 58 dims (96:154) - history=2, flatten (29 joints * 2)
- actions: 29 dims (154:183) - history=1, flatten (29 joints * 1)

Critic obs structure:
- base_lin_vel: 3 dims (0:3)
- base_pos_z: 1 dim (3:4)
- foot_contact: 2 dims (4:6) - left, right foot contact
- base_ang_vel: 3 dims (6:9)
- projected_gravity: 3 dims (9:12)
- velocity_commands: 3 dims (12:15)
- joint_pos: 87 dims (15:102) - history=3, flatten (29 joints * 3)
- joint_vel: 58 dims (102:160) - history=2, flatten (29 joints * 2)
- actions: 29 dims (160:189) - history=1, flatten (29 joints * 1)

G1 Joint Order (29 DOF) - BASED ON CONTROLLED_JOINTS WITH preserve_order=True:
LEG_JOINT_NAMES + WAIST_JOINT_NAMES + ARM_JOINT_NAMES
Left joints come first in each pair.

LEG JOINTS (0-11):
0: left_hip_yaw_joint <-> 1: right_hip_yaw_joint (sign flip)
2: left_hip_roll_joint <-> 3: right_hip_roll_joint (sign flip)
4: left_hip_pitch_joint <-> 5: right_hip_pitch_joint
6: left_knee_joint <-> 7: right_knee_joint
8: left_ankle_pitch_joint <-> 9: right_ankle_pitch_joint
10: left_ankle_roll_joint <-> 11: right_ankle_roll_joint (sign flip)

WAIST JOINTS (12-14):
12: waist_yaw_joint (central, sign flip only)
13: waist_roll_joint (central, sign flip only)
14: waist_pitch_joint (central, no change)

ARM JOINTS (15-28):
15: left_shoulder_pitch_joint <-> 16: right_shoulder_pitch_joint
17: left_shoulder_roll_joint <-> 18: right_shoulder_roll_joint (sign flip)
19: left_shoulder_yaw_joint <-> 20: right_shoulder_yaw_joint (sign flip)
21: left_elbow_joint <-> 22: right_elbow_joint
23: left_wrist_roll_joint <-> 24: right_wrist_roll_joint (sign flip)
25: left_wrist_pitch_joint <-> 26: right_wrist_pitch_joint
27: left_wrist_yaw_joint <-> 28: right_wrist_yaw_joint (sign flip)
"""

import torch


def g1_locomanip_symmetry(env, obs, actions, obs_type="policy"):  # noqa: ARG001
    """G1 humanoid robot symmetry function for loco-manipulation environment.

    Doubles the batch by appending mirrored copies of observations and actions.
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
        mirrored[:, 0] *= -1  # ang_vel_x
        mirrored[:, 2] *= -1  # ang_vel_z

        # projected_gravity (3:6): (x, y, z) -> (x, -y, z)
        mirrored[:, 4] *= -1  # gravity_y

        # velocity_commands (6:9): (vx, vy, vyaw) -> (vx, -vy, -vyaw)
        mirrored[:, 7] *= -1   # vy
        mirrored[:, 8] *= -1   # vyaw

        # joint_pos with history (9:96): 87 dims = 29 joints * 3 history
        mirror_joint_history(mirrored, start_idx=9, joint_dim=29, history_len=3)

        # joint_vel with history (96:154): 58 dims = 29 joints * 2 history
        mirror_joint_history(mirrored, start_idx=96, joint_dim=29, history_len=2)

        # actions with history (154:183): 29 dims = 29 joints * 1 history
        mirror_joint_history(mirrored, start_idx=154, joint_dim=29, history_len=1)

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

        # joint_pos with history (15:102): 87 dims = 29 joints * 3 history
        mirror_joint_history(mirrored, start_idx=15, joint_dim=29, history_len=3)

        # joint_vel with history (102:160): 58 dims = 29 joints * 2 history
        mirror_joint_history(mirrored, start_idx=102, joint_dim=29, history_len=2)

        # actions with history (160:189): 29 dims = 29 joints * 1 history
        mirror_joint_history(mirrored, start_idx=160, joint_dim=29, history_len=1)

    return mirrored


def mirror_joint_history(tensor, start_idx, joint_dim, history_len):
    """Mirror joint data with history for G1 humanoid robot (29 DOF).

    Args:
        tensor: observation tensor
        start_idx: joint data start index
        joint_dim: joint dimension (29)
        history_len: history length (1, 2, or 3)
    """
    for h in range(history_len):
        time_start = start_idx + h * joint_dim

        # Extract joint data for this time step
        joint_data = tensor[:, time_start:time_start + joint_dim].clone()

        # === LEG JOINTS (0-11) ===
        # 0: left_hip_yaw <-> 1: right_hip_yaw (sign flip)
        tensor[:, time_start+0] = joint_data[:, 1]
        tensor[:, time_start+1] = joint_data[:, 0]
        tensor[:, time_start+0] *= -1
        tensor[:, time_start+1] *= -1

        # 2: left_hip_roll <-> 3: right_hip_roll (sign flip)
        tensor[:, time_start+2] = joint_data[:, 3]
        tensor[:, time_start+3] = joint_data[:, 2]
        tensor[:, time_start+2] *= -1
        tensor[:, time_start+3] *= -1

        # 4: left_hip_pitch <-> 5: right_hip_pitch
        tensor[:, time_start+4] = joint_data[:, 5]
        tensor[:, time_start+5] = joint_data[:, 4]

        # 6: left_knee <-> 7: right_knee
        tensor[:, time_start+6] = joint_data[:, 7]
        tensor[:, time_start+7] = joint_data[:, 6]

        # 8: left_ankle_pitch <-> 9: right_ankle_pitch
        tensor[:, time_start+8] = joint_data[:, 9]
        tensor[:, time_start+9] = joint_data[:, 8]

        # 10: left_ankle_roll <-> 11: right_ankle_roll (sign flip)
        tensor[:, time_start+10] = joint_data[:, 11]
        tensor[:, time_start+11] = joint_data[:, 10]
        tensor[:, time_start+10] *= -1
        tensor[:, time_start+11] *= -1

        # === WAIST JOINTS (12-14, central) ===
        tensor[:, time_start+12] *= -1  # waist_yaw: sign flip
        tensor[:, time_start+13] *= -1  # waist_roll: sign flip
        # waist_pitch (14): no change

        # === ARM JOINTS (15-28) ===
        # 15: left_shoulder_pitch <-> 16: right_shoulder_pitch
        tensor[:, time_start+15] = joint_data[:, 16]
        tensor[:, time_start+16] = joint_data[:, 15]

        # 17: left_shoulder_roll <-> 18: right_shoulder_roll (sign flip)
        tensor[:, time_start+17] = joint_data[:, 18]
        tensor[:, time_start+18] = joint_data[:, 17]
        tensor[:, time_start+17] *= -1
        tensor[:, time_start+18] *= -1

        # 19: left_shoulder_yaw <-> 20: right_shoulder_yaw (sign flip)
        tensor[:, time_start+19] = joint_data[:, 20]
        tensor[:, time_start+20] = joint_data[:, 19]
        tensor[:, time_start+19] *= -1
        tensor[:, time_start+20] *= -1

        # 21: left_elbow <-> 22: right_elbow
        tensor[:, time_start+21] = joint_data[:, 22]
        tensor[:, time_start+22] = joint_data[:, 21]

        # 23: left_wrist_roll <-> 24: right_wrist_roll (sign flip)
        tensor[:, time_start+23] = joint_data[:, 24]
        tensor[:, time_start+24] = joint_data[:, 23]
        tensor[:, time_start+23] *= -1
        tensor[:, time_start+24] *= -1

        # 25: left_wrist_pitch <-> 26: right_wrist_pitch
        tensor[:, time_start+25] = joint_data[:, 26]
        tensor[:, time_start+26] = joint_data[:, 25]

        # 27: left_wrist_yaw <-> 28: right_wrist_yaw (sign flip)
        tensor[:, time_start+27] = joint_data[:, 28]
        tensor[:, time_start+28] = joint_data[:, 27]
        tensor[:, time_start+27] *= -1
        tensor[:, time_start+28] *= -1


def mirror_actions(actions):
    """Mirror actions for left-right symmetry transformation (G1 humanoid robot)."""
    mirrored = actions.clone()
    action_temp = mirrored.clone()

    # === LEG JOINTS (0-11) ===
    # 0: left_hip_yaw <-> 1: right_hip_yaw (sign flip)
    mirrored[:, 0] = action_temp[:, 1]
    mirrored[:, 1] = action_temp[:, 0]
    mirrored[:, 0] *= -1
    mirrored[:, 1] *= -1

    # 2: left_hip_roll <-> 3: right_hip_roll (sign flip)
    mirrored[:, 2] = action_temp[:, 3]
    mirrored[:, 3] = action_temp[:, 2]
    mirrored[:, 2] *= -1
    mirrored[:, 3] *= -1

    # 4: left_hip_pitch <-> 5: right_hip_pitch
    mirrored[:, 4] = action_temp[:, 5]
    mirrored[:, 5] = action_temp[:, 4]

    # 6: left_knee <-> 7: right_knee
    mirrored[:, 6] = action_temp[:, 7]
    mirrored[:, 7] = action_temp[:, 6]

    # 8: left_ankle_pitch <-> 9: right_ankle_pitch
    mirrored[:, 8] = action_temp[:, 9]
    mirrored[:, 9] = action_temp[:, 8]

    # 10: left_ankle_roll <-> 11: right_ankle_roll (sign flip)
    mirrored[:, 10] = action_temp[:, 11]
    mirrored[:, 11] = action_temp[:, 10]
    mirrored[:, 10] *= -1
    mirrored[:, 11] *= -1

    # === WAIST JOINTS (12-14, central) ===
    mirrored[:, 12] *= -1  # waist_yaw: sign flip
    mirrored[:, 13] *= -1  # waist_roll: sign flip
    # waist_pitch (14): no change

    # === ARM JOINTS (15-28) ===
    # 15: left_shoulder_pitch <-> 16: right_shoulder_pitch
    mirrored[:, 15] = action_temp[:, 16]
    mirrored[:, 16] = action_temp[:, 15]

    # 17: left_shoulder_roll <-> 18: right_shoulder_roll (sign flip)
    mirrored[:, 17] = action_temp[:, 18]
    mirrored[:, 18] = action_temp[:, 17]
    mirrored[:, 17] *= -1
    mirrored[:, 18] *= -1

    # 19: left_shoulder_yaw <-> 20: right_shoulder_yaw (sign flip)
    mirrored[:, 19] = action_temp[:, 20]
    mirrored[:, 20] = action_temp[:, 19]
    mirrored[:, 19] *= -1
    mirrored[:, 20] *= -1

    # 21: left_elbow <-> 22: right_elbow
    mirrored[:, 21] = action_temp[:, 22]
    mirrored[:, 22] = action_temp[:, 21]

    # 23: left_wrist_roll <-> 24: right_wrist_roll (sign flip)
    mirrored[:, 23] = action_temp[:, 24]
    mirrored[:, 24] = action_temp[:, 23]
    mirrored[:, 23] *= -1
    mirrored[:, 24] *= -1

    # 25: left_wrist_pitch <-> 26: right_wrist_pitch
    mirrored[:, 25] = action_temp[:, 26]
    mirrored[:, 26] = action_temp[:, 25]

    # 27: left_wrist_yaw <-> 28: right_wrist_yaw (sign flip)
    mirrored[:, 27] = action_temp[:, 28]
    mirrored[:, 28] = action_temp[:, 27]
    mirrored[:, 27] *= -1
    mirrored[:, 28] *= -1

    return mirrored
