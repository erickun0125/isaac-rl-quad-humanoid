import torch

def g1_locomanip_symmetry(env, obs, actions, obs_type="policy"):  # noqa: ARG001
    """
    G1 humanoid robot symmetry function for loco-manipulation environment.
    
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
    - base_ang_vel: 3 dims (4:7)
    - projected_gravity: 3 dims (7:10)
    - velocity_commands: 3 dims (10:13)
    - joint_pos: 87 dims (13:100) - history=3, flatten (29 joints * 3)
    - joint_vel: 58 dims (100:158) - history=2, flatten (29 joints * 2)
    - actions: 29 dims (158:187) - history=1, flatten (29 joints * 1)
    
    G1 Joint Order (29 DOF) - BASED ON CONTROLLED_JOINTS WITH preserve_order=True:
    LEG_JOINT_NAMES + WAIST_JOINT_NAMES + ARM_JOINT_NAMES
    Left joints come first in each pair.
    
    LEG JOINTS (0-11):
    0: left_hip_yaw_joint ↔ 1: right_hip_yaw_joint (sign flip)
    2: left_hip_roll_joint ↔ 3: right_hip_roll_joint (sign flip)
    4: left_hip_pitch_joint ↔ 5: right_hip_pitch_joint
    6: left_knee_joint ↔ 7: right_knee_joint
    8: left_ankle_pitch_joint ↔ 9: right_ankle_pitch_joint
    10: left_ankle_roll_joint ↔ 11: right_ankle_roll_joint (sign flip)
    
    WAIST JOINTS (12-14):
    12: waist_yaw_joint (central, sign flip only)
    13: waist_roll_joint (central, sign flip only)
    14: waist_pitch_joint (central, no change)
    
    ARM JOINTS (15-28):
    15: left_shoulder_pitch_joint ↔ 16: right_shoulder_pitch_joint
    17: left_shoulder_roll_joint ↔ 18: right_shoulder_roll_joint (sign flip)
    19: left_shoulder_yaw_joint ↔ 20: right_shoulder_yaw_joint (sign flip)
    21: left_elbow_joint ↔ 22: right_elbow_joint
    23: left_wrist_roll_joint ↔ 24: right_wrist_roll_joint (sign flip)
    25: left_wrist_pitch_joint ↔ 26: right_wrist_pitch_joint
    27: left_wrist_yaw_joint ↔ 28: right_wrist_yaw_joint (sign flip)
    """
    
    # 1. Extend original + mirrored data
    if obs is not None:
        obs_batch_size = obs.shape[0]
        obs_mirrored = torch.cat([obs, obs.clone()], dim=0)
        # Apply symmetry to the second half
        obs_mirrored[obs_batch_size:] = mirror_observations(obs_mirrored[obs_batch_size:], obs_type)
    else:
        obs_mirrored = None
        
    if actions is not None:
        action_batch_size = actions.shape[0]
        actions_mirrored = torch.cat([actions, actions.clone()], dim=0)
        # Apply symmetry to the second half
        actions_mirrored[action_batch_size:] = mirror_actions(actions_mirrored[action_batch_size:])
    else:
        actions_mirrored = None
        
    return obs_mirrored, actions_mirrored


def mirror_observations(obs, obs_type):
    """Mirror observations for left-right symmetry transformation"""
    mirrored = obs.clone()
    
    if obs_type == "policy":
        # Policy observations
        
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
        # Critic observations
        
        # base_lin_vel (0:3): (x, y, z) -> (x, -y, z)
        mirrored[:, 1] *= -1  # lin_vel_y
        
        # base_pos_z (3:4): no change needed
        
        # base_ang_vel (4:7): (x, y, z) -> (-x, y, -z)
        mirrored[:, 4] *= -1  # ang_vel_x
        mirrored[:, 6] *= -1  # ang_vel_z
        
        # projected_gravity (7:10): (x, y, z) -> (x, -y, z)
        mirrored[:, 8] *= -1  # gravity_y
        
        # velocity_commands (10:13): (vx, vy, vyaw) -> (vx, -vy, -vyaw)
        mirrored[:, 11] *= -1  # vy
        mirrored[:, 12] *= -1  # vyaw
        
        # joint_pos with history (13:100): 87 dims = 29 joints * 3 history
        mirror_joint_history(mirrored, start_idx=13, joint_dim=29, history_len=3)
        
        # joint_vel with history (100:158): 58 dims = 29 joints * 2 history
        mirror_joint_history(mirrored, start_idx=100, joint_dim=29, history_len=2)
        
        # actions with history (158:187): 29 dims = 29 joints * 1 history
        mirror_joint_history(mirrored, start_idx=158, joint_dim=29, history_len=1)
    
    return mirrored


def mirror_joint_history(tensor, start_idx, joint_dim, history_len):
    """
    Mirror joint data with history for G1 humanoid robot (29 DOF)
    
    Args:
        tensor: observation tensor
        start_idx: joint data start index
        joint_dim: joint dimension (29)
        history_len: history length (1, 2, or 3)
    """
    # History is flattened as [t-history+1, t-history+2, ..., t]
    for h in range(history_len):
        # Mirror joints for each time step
        time_start = start_idx + h * joint_dim
        time_end = time_start + joint_dim
        
        # Extract joint data for this time step
        joint_data = tensor[:, time_start:time_end].clone()
        
        # G1 joint mirroring (29 DOF total) - BASED ON CONTROLLED_JOINTS ORDER:
        # LEG_JOINT_NAMES + WAIST_JOINT_NAMES + ARM_JOINT_NAMES
        # Left joints come first in each pair
        
        # === LEG JOINTS (0-11) ===
        # 0: left_hip_yaw ↔ 1: right_hip_yaw (sign flip)
        tensor[:, time_start+0] = joint_data[:, 1]   # left_hip_yaw = right_hip_yaw
        tensor[:, time_start+1] = joint_data[:, 0]   # right_hip_yaw = left_hip_yaw
        tensor[:, time_start+0] *= -1  # left_hip_yaw (now right)
        tensor[:, time_start+1] *= -1  # right_hip_yaw (now left)
        
        # 2: left_hip_roll ↔ 3: right_hip_roll (sign flip)
        tensor[:, time_start+2] = joint_data[:, 3]   # left_hip_roll = right_hip_roll
        tensor[:, time_start+3] = joint_data[:, 2]   # right_hip_roll = left_hip_roll
        tensor[:, time_start+2] *= -1  # left_hip_roll (now right)
        tensor[:, time_start+3] *= -1  # right_hip_roll (now left)
        
        # 4: left_hip_pitch ↔ 5: right_hip_pitch
        tensor[:, time_start+4] = joint_data[:, 5]   # left_hip_pitch = right_hip_pitch
        tensor[:, time_start+5] = joint_data[:, 4]   # right_hip_pitch = left_hip_pitch
        
        # 6: left_knee ↔ 7: right_knee
        tensor[:, time_start+6] = joint_data[:, 7]   # left_knee = right_knee
        tensor[:, time_start+7] = joint_data[:, 6]   # right_knee = left_knee
        
        # 8: left_ankle_pitch ↔ 9: right_ankle_pitch
        tensor[:, time_start+8] = joint_data[:, 9]   # left_ankle_pitch = right_ankle_pitch
        tensor[:, time_start+9] = joint_data[:, 8]   # right_ankle_pitch = left_ankle_pitch
        
        # 10: left_ankle_roll ↔ 11: right_ankle_roll (sign flip)
        tensor[:, time_start+10] = joint_data[:, 11]  # left_ankle_roll = right_ankle_roll
        tensor[:, time_start+11] = joint_data[:, 10]  # right_ankle_roll = left_ankle_roll
        tensor[:, time_start+10] *= -1  # left_ankle_roll (now right)
        tensor[:, time_start+11] *= -1  # right_ankle_roll (now left)
        
        # === WAIST JOINTS (12-14, central) ===
        # 12: waist_yaw_joint - sign flip only
        tensor[:, time_start+12] *= -1
        
        # 13: waist_roll_joint - sign flip only
        tensor[:, time_start+13] *= -1
        
        # 14: waist_pitch_joint - no change
        
        # === ARM JOINTS (15-28) ===
        # 15: left_shoulder_pitch ↔ 16: right_shoulder_pitch
        tensor[:, time_start+15] = joint_data[:, 16]  # left_shoulder_pitch = right_shoulder_pitch
        tensor[:, time_start+16] = joint_data[:, 15]  # right_shoulder_pitch = left_shoulder_pitch
        
        # 17: left_shoulder_roll ↔ 18: right_shoulder_roll (sign flip)
        tensor[:, time_start+17] = joint_data[:, 18]  # left_shoulder_roll = right_shoulder_roll
        tensor[:, time_start+18] = joint_data[:, 17]  # right_shoulder_roll = left_shoulder_roll
        tensor[:, time_start+17] *= -1  # left_shoulder_roll (now right)
        tensor[:, time_start+18] *= -1  # right_shoulder_roll (now left)
        
        # 19: left_shoulder_yaw ↔ 20: right_shoulder_yaw (sign flip)
        tensor[:, time_start+19] = joint_data[:, 20]  # left_shoulder_yaw = right_shoulder_yaw
        tensor[:, time_start+20] = joint_data[:, 19]  # right_shoulder_yaw = left_shoulder_yaw
        tensor[:, time_start+19] *= -1  # left_shoulder_yaw (now right)
        tensor[:, time_start+20] *= -1  # right_shoulder_yaw (now left)
        
        # 21: left_elbow ↔ 22: right_elbow
        tensor[:, time_start+21] = joint_data[:, 22]  # left_elbow = right_elbow
        tensor[:, time_start+22] = joint_data[:, 21]  # right_elbow = left_elbow
        
        # 23: left_wrist_roll ↔ 24: right_wrist_roll (sign flip)
        tensor[:, time_start+23] = joint_data[:, 24]  # left_wrist_roll = right_wrist_roll
        tensor[:, time_start+24] = joint_data[:, 23]  # right_wrist_roll = left_wrist_roll
        tensor[:, time_start+23] *= -1  # left_wrist_roll (now right)
        tensor[:, time_start+24] *= -1  # right_wrist_roll (now left)
        
        # 25: left_wrist_pitch ↔ 26: right_wrist_pitch
        tensor[:, time_start+25] = joint_data[:, 26]  # left_wrist_pitch = right_wrist_pitch
        tensor[:, time_start+26] = joint_data[:, 25]  # right_wrist_pitch = left_wrist_pitch
        
        # 27: left_wrist_yaw ↔ 28: right_wrist_yaw (sign flip)
        tensor[:, time_start+27] = joint_data[:, 28]  # left_wrist_yaw = right_wrist_yaw
        tensor[:, time_start+28] = joint_data[:, 27]  # right_wrist_yaw = left_wrist_yaw
        tensor[:, time_start+27] *= -1  # left_wrist_yaw (now right)
        tensor[:, time_start+28] *= -1  # right_wrist_yaw (now left)


def mirror_actions(actions):
    """Mirror actions for left-right symmetry transformation (G1 humanoid robot)"""
    mirrored = actions.clone()
    
    # G1 joint actions (29 DOF) - BASED ON CONTROLLED_JOINTS ORDER
    action_temp = mirrored.clone()
    
    # === LEG JOINTS (0-11) ===
    # 0: left_hip_yaw ↔ 1: right_hip_yaw (sign flip)
    mirrored[:, 0] = action_temp[:, 1]   # left_hip_yaw = right_hip_yaw
    mirrored[:, 1] = action_temp[:, 0]   # right_hip_yaw = left_hip_yaw
    mirrored[:, 0] *= -1  # left_hip_yaw (now right)
    mirrored[:, 1] *= -1  # right_hip_yaw (now left)
    
    # 2: left_hip_roll ↔ 3: right_hip_roll (sign flip)
    mirrored[:, 2] = action_temp[:, 3]   # left_hip_roll = right_hip_roll
    mirrored[:, 3] = action_temp[:, 2]   # right_hip_roll = left_hip_roll
    mirrored[:, 2] *= -1  # left_hip_roll (now right)
    mirrored[:, 3] *= -1  # right_hip_roll (now left)
    
    # 4: left_hip_pitch ↔ 5: right_hip_pitch
    mirrored[:, 4] = action_temp[:, 5]   # left_hip_pitch = right_hip_pitch
    mirrored[:, 5] = action_temp[:, 4]   # right_hip_pitch = left_hip_pitch
    
    # 6: left_knee ↔ 7: right_knee
    mirrored[:, 6] = action_temp[:, 7]   # left_knee = right_knee
    mirrored[:, 7] = action_temp[:, 6]   # right_knee = left_knee
    
    # 8: left_ankle_pitch ↔ 9: right_ankle_pitch
    mirrored[:, 8] = action_temp[:, 9]   # left_ankle_pitch = right_ankle_pitch
    mirrored[:, 9] = action_temp[:, 8]   # right_ankle_pitch = left_ankle_pitch
    
    # 10: left_ankle_roll ↔ 11: right_ankle_roll (sign flip)
    mirrored[:, 10] = action_temp[:, 11]  # left_ankle_roll = right_ankle_roll
    mirrored[:, 11] = action_temp[:, 10]  # right_ankle_roll = left_ankle_roll
    mirrored[:, 10] *= -1  # left_ankle_roll (now right)
    mirrored[:, 11] *= -1  # right_ankle_roll (now left)
    
    # === WAIST JOINTS (12-14, central) ===
    # 12: waist_yaw_joint - sign flip only
    mirrored[:, 12] *= -1
    
    # 13: waist_roll_joint - sign flip only
    mirrored[:, 13] *= -1
    
    # 14: waist_pitch_joint - no change
    
    # === ARM JOINTS (15-28) ===
    # 15: left_shoulder_pitch ↔ 16: right_shoulder_pitch
    mirrored[:, 15] = action_temp[:, 16]  # left_shoulder_pitch = right_shoulder_pitch
    mirrored[:, 16] = action_temp[:, 15]  # right_shoulder_pitch = left_shoulder_pitch
    
    # 17: left_shoulder_roll ↔ 18: right_shoulder_roll (sign flip)
    mirrored[:, 17] = action_temp[:, 18]  # left_shoulder_roll = right_shoulder_roll
    mirrored[:, 18] = action_temp[:, 17]  # right_shoulder_roll = left_shoulder_roll
    mirrored[:, 17] *= -1  # left_shoulder_roll (now right)
    mirrored[:, 18] *= -1  # right_shoulder_roll (now left)
    
    # 19: left_shoulder_yaw ↔ 20: right_shoulder_yaw (sign flip)
    mirrored[:, 19] = action_temp[:, 20]  # left_shoulder_yaw = right_shoulder_yaw
    mirrored[:, 20] = action_temp[:, 19]  # right_shoulder_yaw = left_shoulder_yaw
    mirrored[:, 19] *= -1  # left_shoulder_yaw (now right)
    mirrored[:, 20] *= -1  # right_shoulder_yaw (now left)
    
    # 21: left_elbow ↔ 22: right_elbow
    mirrored[:, 21] = action_temp[:, 22]  # left_elbow = right_elbow
    mirrored[:, 22] = action_temp[:, 21]  # right_elbow = left_elbow
    
    # 23: left_wrist_roll ↔ 24: right_wrist_roll (sign flip)
    mirrored[:, 23] = action_temp[:, 24]  # left_wrist_roll = right_wrist_roll
    mirrored[:, 24] = action_temp[:, 23]  # right_wrist_roll = left_wrist_roll
    mirrored[:, 23] *= -1  # left_wrist_roll (now right)
    mirrored[:, 24] *= -1  # right_wrist_roll (now left)
    
    # 25: left_wrist_pitch ↔ 26: right_wrist_pitch
    mirrored[:, 25] = action_temp[:, 26]  # left_wrist_pitch = right_wrist_pitch
    mirrored[:, 26] = action_temp[:, 25]  # right_wrist_pitch = left_wrist_pitch
    
    # 27: left_wrist_yaw ↔ 28: right_wrist_yaw (sign flip)
    mirrored[:, 27] = action_temp[:, 28]  # left_wrist_yaw = right_wrist_yaw
    mirrored[:, 28] = action_temp[:, 27]  # right_wrist_yaw = left_wrist_yaw
    mirrored[:, 27] *= -1  # left_wrist_yaw (now right)
    mirrored[:, 28] *= -1  # right_wrist_yaw (now left)
    
    return mirrored


# Usage example and testing
if __name__ == "__main__":
    # Policy observation test (estimated dimensions for G1 loco-manipulation)
    batch_size = 4
    policy_obs_dim = 183  # 3 + 3 + 3 + 87 + 58 + 29
    policy_obs = torch.randn(batch_size, policy_obs_dim)
    test_actions = torch.randn(batch_size, 29)  # 29 DOF for G1
    
    # Symmetry transformation test
    mirrored_obs, mirrored_actions = g1_locomanip_symmetry(
        env=None, obs=policy_obs, actions=test_actions, obs_type="policy"
    )
    
    print(f"Original policy obs shape: {policy_obs.shape}")
    print(f"Mirrored policy obs shape: {mirrored_obs.shape}")
    print(f"Original actions shape: {test_actions.shape}")
    print(f"Mirrored actions shape: {mirrored_actions.shape}")
    
    # Critic observation test (estimated dimensions)
    critic_obs_dim = 187  # 3 + 1 + 3 + 3 + 3 + 87 + 58 + 29
    critic_obs = torch.randn(batch_size, critic_obs_dim)
    mirrored_critic_obs, _ = g1_locomanip_symmetry(
        env=None, obs=critic_obs, actions=None, obs_type="critic"
    )
    
    print(f"Original critic obs shape: {critic_obs.shape}")
    print(f"Mirrored critic obs shape: {mirrored_critic_obs.shape}")
    
    print("\n=== G1 Humanoid Robot Joint Symmetry Mapping - UPDATED ===")
    print("Based on CONTROLLED_JOINTS = LEG_JOINT_NAMES + WAIST_JOINT_NAMES + ARM_JOINT_NAMES")
    print("Left joints come first in each pair (preserve_order=True)")
    print("\nLEG JOINTS:")
    print("  left_hip_yaw (0) ↔ right_hip_yaw (1), sign flip")
    print("  left_hip_roll (2) ↔ right_hip_roll (3), sign flip")
    print("  left_hip_pitch (4) ↔ right_hip_pitch (5)")
    print("  left_knee (6) ↔ right_knee (7)")
    print("  left_ankle_pitch (8) ↔ right_ankle_pitch (9)")
    print("  left_ankle_roll (10) ↔ right_ankle_roll (11), sign flip")
    print("\nWAIST JOINTS (central):")
    print("  waist_yaw_joint (12) - sign flip only")
    print("  waist_roll_joint (13) - sign flip only")
    print("  waist_pitch_joint (14) - no change")
    print("\nARM JOINTS:")
    print("  left_shoulder_pitch (15) ↔ right_shoulder_pitch (16)")
    print("  left_shoulder_roll (17) ↔ right_shoulder_roll (18), sign flip")
    print("  left_shoulder_yaw (19) ↔ right_shoulder_yaw (20), sign flip")
    print("  left_elbow (21) ↔ right_elbow (22)")
    print("  left_wrist_roll (23) ↔ right_wrist_roll (24), sign flip")
    print("  left_wrist_pitch (25) ↔ right_wrist_pitch (26)")
    print("  left_wrist_yaw (27) ↔ right_wrist_yaw (28), sign flip")
    
    # Training configuration usage
    print("\n=== Training Configuration ===")
    print("""
    agent_cfg.algorithm.symmetry_cfg = {
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "data_augmentation_func": g1_locomanip_symmetry,
        "mirror_loss_coeff": 0.1,
    }
    
    # UPDATED Joint Order from CONTROLLED_JOINTS with preserve_order=True:
    # LEG_JOINT_NAMES + WAIST_JOINT_NAMES + ARM_JOINT_NAMES
    # Left joints come first in each pair
    # Total: 29 DOF (6*2 + 3 + 7*2 = 12 + 3 + 14 = 29)
    # 
    # Note: Adjust observation dimensions based on actual environment setup:
    # - Policy obs: base_ang_vel(3) + projected_gravity(3) + velocity_commands(3) + 
    #               joint_pos_history(29*3) + joint_vel_history(29*2) + action_history(29*1)
    # - Critic obs: base_lin_vel(3) + base_pos_z(1) + base_ang_vel(3) + projected_gravity(3) +
    #               velocity_commands(3) + joint_pos_history(29*3) + joint_vel_history(29*2) + action_history(29*1)
    """)