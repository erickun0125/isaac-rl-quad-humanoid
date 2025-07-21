import torch

def custom_locomotion_symmetry(env, obs, actions, obs_type="policy"):
    """
    Go2 로봇의 observation 구조에 맞는 대칭 함수.
    
    Policy obs structure (93차원):
    - base_ang_vel: 3차원 (0:3)
    - projected_gravity: 3차원 (3:6) 
    - velocity_commands: 3차원 (6:9)
    - joint_pos: 36차원 (9:45) - history=3, flatten
    - joint_vel: 24차원 (45:69) - history=2, flatten
    - actions: 24차원 (69:93) - history=2, flatten
    
    Critic obs structure (96차원):
    - base_lin_vel: 3차원 (0:3)
    - base_ang_vel: 3차원 (3:6)
    - projected_gravity: 3차원 (6:9)
    - velocity_commands: 3차원 (9:12)
    - joint_pos: 36차원 (12:48) - history=3, flatten
    - joint_vel: 24차원 (48:72) - history=2, flatten
    - actions: 24차원 (72:96) - history=2, flatten
    
    Go2 Joint Order:
    [FL_hip, FR_hip, RL_hip, RR_hip,
     FL_thigh, FR_thigh, RL_thigh, RR_thigh,
     FL_calf, FR_calf, RL_calf, RR_calf]
    """
    
    # 1. 원본 + 대칭 데이터로 확장
    if obs is not None:
        batch_size = obs.shape[0]
        obs_mirrored = torch.cat([obs, obs.clone()], dim=0)
        # 두 번째 절반에 대칭 적용
        obs_mirrored[batch_size:] = mirror_observations(obs_mirrored[batch_size:], obs_type)
    else:
        obs_mirrored = None
        
    if actions is not None:
        batch_size = actions.shape[0]
        actions_mirrored = torch.cat([actions, actions.clone()], dim=0)
        # 두 번째 절반에 대칭 적용
        actions_mirrored[batch_size:] = mirror_actions(actions_mirrored[batch_size:])
    else:
        actions_mirrored = None
        
    return obs_mirrored, actions_mirrored


def mirror_observations(obs, obs_type):
    """Observation을 좌우 대칭으로 변환"""
    mirrored = obs.clone()
    
    if obs_type == "policy":
        # Policy observations (93차원)
        
        # base_ang_vel (0:3): (x, y, z) -> (-x, y, -z)
        mirrored[:, 0] *= -1  # ang_vel_x
        mirrored[:, 2] *= -1  # ang_vel_z
        
        # projected_gravity (3:6): (x, y, z) -> (x, -y, z)
        mirrored[:, 4] *= -1  # gravity_y
        
        # velocity_commands (6:9): (vx, vy, vyaw) -> (vx, -vy, -vyaw)
        mirrored[:, 7] *= -1   # vy
        mirrored[:, 8] *= -1   # vyaw
        
        # joint_pos with history (9:45): 36차원 = 12joints * 3history
        mirror_joint_history(mirrored, start_idx=9, joint_dim=12, history_len=3)
        
        # joint_vel with history (45:69): 24차원 = 12joints * 2history  
        mirror_joint_history(mirrored, start_idx=45, joint_dim=12, history_len=2)
        
        # actions with history (69:93): 24차원 = 12joints * 2history
        mirror_joint_history(mirrored, start_idx=69, joint_dim=12, history_len=2)
        
    elif obs_type == "critic":
        # Critic observations (96차원)
        
        # base_lin_vel (0:3): (x, y, z) -> (x, -y, z)
        mirrored[:, 1] *= -1  # lin_vel_y
        
        # base_ang_vel (3:6): (x, y, z) -> (-x, y, -z)
        mirrored[:, 3] *= -1  # ang_vel_x
        mirrored[:, 5] *= -1  # ang_vel_z
        
        # projected_gravity (6:9): (x, y, z) -> (x, -y, z)
        mirrored[:, 7] *= -1  # gravity_y
        
        # velocity_commands (9:12): (vx, vy, vyaw) -> (vx, -vy, -vyaw)
        mirrored[:, 10] *= -1  # vy
        mirrored[:, 11] *= -1  # vyaw
        
        # joint_pos with history (12:48): 36차원 = 12joints * 3history
        mirror_joint_history(mirrored, start_idx=12, joint_dim=12, history_len=3)
        
        # joint_vel with history (48:72): 24차원 = 12joints * 2history
        mirror_joint_history(mirrored, start_idx=48, joint_dim=12, history_len=2)
        
        # actions with history (72:96): 24차원 = 12joints * 2history
        mirror_joint_history(mirrored, start_idx=72, joint_dim=12, history_len=2)
    
    return mirrored


def mirror_joint_history(tensor, start_idx, joint_dim, history_len):
    """
    History가 flatten된 joint data를 대칭 변환 (Go2 로봇용)
    
    Args:
        tensor: observation tensor
        start_idx: joint data 시작 인덱스
        joint_dim: joint 차원 (12)
        history_len: history 길이 (2 or 3)
    """
    # History는 [t-history+1, t-history+2, ..., t] 순서로 flatten됨
    for h in range(history_len):
        # 각 time step별로 joint 대칭 변환
        time_start = start_idx + h * joint_dim
        time_end = time_start + joint_dim
        
        # 해당 time step의 joint data 추출
        joint_data = tensor[:, time_start:time_end].clone()
        
        # Go2 joint mirroring: FL<->FR, RL<->RR
        # Go2 joints: [FL_hip, FR_hip, RL_hip, RR_hip,
        #              FL_thigh, FR_thigh, RL_thigh, RR_thigh,
        #              FL_calf, FR_calf, RL_calf, RR_calf]
        
        # Hip joints (0-3): FL<->FR, RL<->RR
        tensor[:, time_start+0] = joint_data[:, 1]   # FL_hip = FR_hip
        tensor[:, time_start+1] = joint_data[:, 0]   # FR_hip = FL_hip
        tensor[:, time_start+2] = joint_data[:, 3]   # RL_hip = RR_hip
        tensor[:, time_start+3] = joint_data[:, 2]   # RR_hip = RL_hip
        
        # Thigh joints (4-7): FL<->FR, RL<->RR
        tensor[:, time_start+4] = joint_data[:, 5]   # FL_thigh = FR_thigh
        tensor[:, time_start+5] = joint_data[:, 4]   # FR_thigh = FL_thigh
        tensor[:, time_start+6] = joint_data[:, 7]   # RL_thigh = RR_thigh
        tensor[:, time_start+7] = joint_data[:, 6]   # RR_thigh = RL_thigh
        
        # Calf joints (8-11): FL<->FR, RL<->RR
        tensor[:, time_start+8] = joint_data[:, 9]    # FL_calf = FR_calf
        tensor[:, time_start+9] = joint_data[:, 8]    # FR_calf = FL_calf
        tensor[:, time_start+10] = joint_data[:, 11]  # RL_calf = RR_calf
        tensor[:, time_start+11] = joint_data[:, 10]  # RR_calf = RL_calf
        
        # Hip 관절 부호 변경 (좌우 회전)
        tensor[:, time_start+0] *= -1   # FL_hip (now FR_hip)
        tensor[:, time_start+1] *= -1   # FR_hip (now FL_hip)
        tensor[:, time_start+2] *= -1   # RL_hip (now RR_hip)
        tensor[:, time_start+3] *= -1   # RR_hip (now RL_hip)


def mirror_actions(actions):
    """Action을 좌우 대칭으로 변환 (Go2 로봇용)"""
    mirrored = actions.clone()
    
    # Go2 joint actions: [FL_hip, FR_hip, RL_hip, RR_hip,
    #                     FL_thigh, FR_thigh, RL_thigh, RR_thigh,
    #                     FL_calf, FR_calf, RL_calf, RR_calf]
    
    action_temp = mirrored.clone()
    
    # Hip joints (0-3): FL<->FR, RL<->RR
    mirrored[:, 0] = action_temp[:, 1]   # FL_hip = FR_hip
    mirrored[:, 1] = action_temp[:, 0]   # FR_hip = FL_hip
    mirrored[:, 2] = action_temp[:, 3]   # RL_hip = RR_hip
    mirrored[:, 3] = action_temp[:, 2]   # RR_hip = RL_hip
    
    # Thigh joints (4-7): FL<->FR, RL<->RR
    mirrored[:, 4] = action_temp[:, 5]   # FL_thigh = FR_thigh
    mirrored[:, 5] = action_temp[:, 4]   # FR_thigh = FL_thigh
    mirrored[:, 6] = action_temp[:, 7]   # RL_thigh = RR_thigh
    mirrored[:, 7] = action_temp[:, 6]   # RR_thigh = RL_thigh
    
    # Calf joints (8-11): FL<->FR, RL<->RR
    mirrored[:, 8] = action_temp[:, 9]    # FL_calf = FR_calf
    mirrored[:, 9] = action_temp[:, 8]    # FR_calf = FL_calf
    mirrored[:, 10] = action_temp[:, 11]  # RL_calf = RR_calf
    mirrored[:, 11] = action_temp[:, 10]  # RR_calf = RL_calf
    
    # Hip 관절 부호 변경 (좌우 회전)
    mirrored[:, 0] *= -1   # FL_hip (now FR_hip)
    mirrored[:, 1] *= -1   # FR_hip (now FL_hip)
    mirrored[:, 2] *= -1   # RL_hip (now RR_hip)
    mirrored[:, 3] *= -1   # RR_hip (now RL_hip)
    
    return mirrored


# 사용 예시 및 테스트
if __name__ == "__main__":
    # Policy observation 테스트 (93차원)
    batch_size = 4
    policy_obs = torch.randn(batch_size, 93)
    actions = torch.randn(batch_size, 12)
    
    # 대칭 변환 테스트
    mirrored_obs, mirrored_actions = custom_locomotion_symmetry(
        env=None, obs=policy_obs, actions=actions, obs_type="policy"
    )
    
    print(f"Original policy obs shape: {policy_obs.shape}")
    print(f"Mirrored policy obs shape: {mirrored_obs.shape}")
    print(f"Original actions shape: {actions.shape}")
    print(f"Mirrored actions shape: {mirrored_actions.shape}")
    
    # Critic observation 테스트 (96차원)
    critic_obs = torch.randn(batch_size, 96)
    mirrored_critic_obs, _ = custom_locomotion_symmetry(
        env=None, obs=critic_obs, actions=None, obs_type="critic"
    )
    
    print(f"Original critic obs shape: {critic_obs.shape}")
    print(f"Mirrored critic obs shape: {mirrored_critic_obs.shape}")
    
    print("\n=== Go2 Robot Joint Symmetry Mapping ===")
    print("FL_hip (0) ↔ FR_hip (1), sign flip")
    print("RL_hip (2) ↔ RR_hip (3), sign flip") 
    print("FL_thigh (4) ↔ FR_thigh (5)")
    print("RL_thigh (6) ↔ RR_thigh (7)")
    print("FL_calf (8) ↔ FR_calf (9)")
    print("RL_calf (10) ↔ RR_calf (11)")
    
    # Training configuration에서 사용하는 방법
    print("\n=== Training Configuration ===")
    print("""
    agent_cfg.algorithm.symmetry_cfg = {
        "use_data_augmentation": True,
        "use_mirror_loss": True,
        "data_augmentation_func": custom_locomotion_symmetry,
        "mirror_loss_coeff": 0.1,
    }
    """)