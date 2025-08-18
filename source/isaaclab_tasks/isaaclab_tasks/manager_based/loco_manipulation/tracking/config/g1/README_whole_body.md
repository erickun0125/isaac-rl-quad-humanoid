# G1 Whole Body Control Environment

이 환경은 Unitree G1 휴머노이드 로봇의 유연한 Whole Body Control을 위한 강화학습 환경입니다.

## 🚀 핵심 특징

### 4개 관절 그룹
- **Hand**: 손가락 관절 (DEX3 hands, 7 DOF per hand = 14 DOF total)
- **Arm**: 팔 관절 (7 DOF per arm = 14 DOF total)  
- **Waist**: 허리 관절 (3 DOF)
- **Leg**: 다리 관절 (6 DOF per leg = 12 DOF total)

### 3가지 정책 타입 (그룹별 설정 가능)
- **RL Policy**: 강화학습으로 제어
- **IL Policy**: 모방학습으로 제어  
- **IK Policy**: 역기구학 솔버로 제어 (Pink IK 또는 Simple IK)

### 유연한 제어 구성
- **Upper Body** = Hand + Arm (28 DOF)
- **Lower Body** = Waist + Leg (15 DOF)
- 각 그룹별로 독립적인 정책 타입 설정 가능

## 📁 파일 구조

```
g1/
├── whole_body_env_cfg.py           # 메인 whole body 환경 설정
├── upper_body_IK.py               # IK 컨트롤러 모듈
├── upper_body_IL.py               # IL 컨트롤러 모듈  
├── mdp/
│   ├── __init__.py
│   ├── lower_body_actions.py      # 기존 lower body action 클래스
│   └── whole_body_actions.py      # 새로운 whole body action 클래스
├── agents/
│   ├── lower_body_rsl_rl_ppo_cfg.py
│   └── whole_body_rsl_rl_ppo_cfg.py # Whole body PPO 설정
├── test_whole_body_env.py         # 테스트 스크립트
└── README_whole_body.md           # 이 파일
```

## 🎯 등록된 환경

### 기본 환경
- **`Isaac-Tracking-WholeBody-G1-v0`**: 기본 whole body 환경 (Hand+Arm=IK, Waist+Leg=RL)
- **`Isaac-Tracking-WholeBody-G1-Play-v0`**: 플레이/테스트용 (50 envs)

### 특화 환경
- **`Isaac-Tracking-WholeBody-G1-UpperBodyIK-v0`**: Upper body는 IK, Lower body는 RL
- **`Isaac-Tracking-WholeBody-G1-UpperBodyIL-v0`**: Upper body는 IL, Lower body는 RL
- **`Isaac-Tracking-WholeBody-G1-FullRL-v0`**: 모든 관절 RL 제어

**모든 환경은 base_velocity 명령만 사용하며, end-effector 제어는 IK/IL을 통해 내부적으로 처리됩니다.**



## 🔧 정책 설정 예시

### 1. Lower Body RL + Upper Body IK (Pink IK 사용)
```python
# G1WholeBodyActionsCfg에서 설정
joint_pos = g1_mdp.WholeBodyJointPositionActionCfg(
    hand_policy=g1_mdp.PolicyType.IK,    # IK 제어 (Hand는 0으로 설정)
    arm_policy=g1_mdp.PolicyType.IK,     # IK 제어 (Pink IK 사용)
    waist_policy=g1_mdp.PolicyType.RL,   # RL 제어  
    leg_policy=g1_mdp.PolicyType.RL,     # RL 제어
    # Pink IK 설정 (선택사항)
    urdf_path="/path/to/g1.urdf",        # G1 URDF 파일 경로
    mesh_path="/path/to/meshes/",        # 메시 파일 경로 (선택사항)
)
# RL Action Dimension: Waist(3) + Leg(12) = 15 DOF
```

### 2. Upper Body IL + Lower Body RL  
```python
joint_pos = g1_mdp.WholeBodyJointPositionActionCfg(
    hand_policy=g1_mdp.PolicyType.IL,    # IL 제어
    arm_policy=g1_mdp.PolicyType.IL,     # IL 제어
    waist_policy=g1_mdp.PolicyType.RL,   # RL 제어
    leg_policy=g1_mdp.PolicyType.RL,     # RL 제어
)
# RL Action Dimension: Waist(3) + Leg(12) = 15 DOF
```

### 3. Full RL Control
```python
joint_pos = g1_mdp.WholeBodyJointPositionActionCfg(
    hand_policy=g1_mdp.PolicyType.RL,    # RL 제어
    arm_policy=g1_mdp.PolicyType.RL,     # RL 제어
    waist_policy=g1_mdp.PolicyType.RL,   # RL 제어
    leg_policy=g1_mdp.PolicyType.RL,     # RL 제어
)
# RL Action Dimension: Hand(14) + Arm(14) + Waist(3) + Leg(12) = 43 DOF
```

## 🔧 Pink IK 설정 (선택사항)

Pink IK를 사용하려면 다음과 같이 설정하세요:

### 1. Pink IK 설치
```bash
pip install pink-python
```

### 2. URDF 파일 준비
G1 로봇의 URDF 파일이 필요합니다. 환경 설정에서 URDF 경로를 지정하세요:

```python
@configclass
class MyCustomActionsCfg(G1WholeBodyActionsCfg):
    joint_pos = g1_mdp.WholeBodyJointPositionActionCfg(
        # ... 기타 설정 ...
        urdf_path="/path/to/your/g1_robot.urdf",  # G1 URDF 파일 경로
        mesh_path="/path/to/meshes/",             # 메시 파일 경로 (선택사항)
    )
```

### 3. Fallback 동작
- URDF 경로가 제공되지 않으면 Simple IK 사용
- Pink IK 초기화 실패 시 자동으로 Simple IK로 fallback
- Hand joint는 Pink IK 사용 여부와 관계없이 모두 0으로 설정

## 🚀 사용 방법

### 1. 기본 사용법
```python
import gymnasium as gym
import isaaclab_tasks

# Lower body RL + Upper body IK 환경
env = gym.make("Isaac-Tracking-WholeBody-G1-LowerBodyRL-v0", num_envs=64)

# 환경 정보 확인
print(f"RL Action dimension: {env.action_space}")
print(f"Observation spaces: {list(env.observation_space.keys())}")

# 시뮬레이션
obs, _ = env.reset()
for step in range(1000):
    # RL 정책이 제어하는 관절에 대한 액션만 필요
    actions = torch.rand(env.action_space.shape, device=env.device) * 0.2 - 0.1
    obs, rewards, terminated, truncated, info = env.step(actions)
    
env.close()
```

### 2. 테스트 스크립트 실행
```bash
# Lower body RL 환경 테스트
python test_whole_body_env.py --env Isaac-Tracking-WholeBody-G1-LowerBodyRL-v0 --num_envs 64

# Full RL 환경 테스트  
python test_whole_body_env.py --env Isaac-Tracking-WholeBody-G1-FullRL-v0 --num_envs 32

# Upper body IL 환경 테스트
python test_whole_body_env.py --env Isaac-Tracking-WholeBody-G1-UpperBodyIL-v0 --num_envs 64
```

### 3. 훈련 실행
```bash
# Lower body RL 훈련
python scripts/rsl_rl/train.py --task Isaac-Tracking-WholeBody-G1-LowerBodyRL-v0 --num_envs 4096

# Full RL 훈련 (더 복잡하므로 적은 환경 수)
python scripts/rsl_rl/train.py --task Isaac-Tracking-WholeBody-G1-FullRL-v0 --num_envs 2048

# Upper body IL 훈련
python scripts/rsl_rl/train.py --task Isaac-Tracking-WholeBody-G1-UpperBodyIL-v0 --num_envs 4096
```

## 🔍 내부 동작 원리

### Action Processing Flow
1. **RL Policy Actions**: `env.step(actions)`에서 RL-controlled joints만 전달
2. **IK Controller**: 시뮬레이션 시간 기반으로 Cartesian trajectory 생성 → IK 해결 → joint targets
3. **IL Controller**: 현재 observations → pre-trained IL model → joint targets  
4. **Action Combination**: 모든 그룹의 targets 결합 → robot에 전달

### Policy Type별 특징
- **RL**: `env.step()`의 actions에서 오는 학습 가능한 정책
- **IK**: 미리 정의된 Cartesian trajectory를 따르는 deterministic 제어
  - **Pink IK**: 고급 미분 가능한 IK 솔버 (URDF 경로 제공 시 사용)
  - **Simple IK**: 기하학적 IK 솔버 (fallback)
  - **Hand**: 모든 joint target을 0으로 설정
- **IL**: Pre-trained model을 사용한 모방학습 기반 제어

### Observation Space
- **Policy Network**: 노이즈 포함, 모든 joint states 관찰 (awareness)
- **Critic Network**: 특권 정보 포함, 노이즈 없는 관찰
- **Action History**: RL-controlled joints만 포함

## 🛠️ 커스터마이제이션

### 1. 새로운 정책 조합 만들기
```python
@configclass
class MyCustomEnvCfg(G1WholeBodyEnvCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        # 원하는 정책 조합으로 설정
        self.actions.joint_pos.hand_policy = g1_mdp.PolicyType.RL
        self.actions.joint_pos.arm_policy = g1_mdp.PolicyType.IK
        self.actions.joint_pos.waist_policy = g1_mdp.PolicyType.IL
        self.actions.joint_pos.leg_policy = g1_mdp.PolicyType.RL
```

### 2. IK Controller 수정
```python
# upper_body_IK.py에서 수정
class CustomTrajectoryGenerator(TrajectoryGenerator):
    def generate(self, current_time: float, **kwargs) -> Dict[str, torch.Tensor]:
        # 커스텀 trajectory 구현
        pass
```

### 3. IL Model 교체
```python
# upper_body_IL.py에서 수정
class CustomILModel(ILModel):
    def predict(self, observations: torch.Tensor) -> torch.Tensor:
        # 커스텀 IL model 구현
        pass
```

### 4. 보상 함수 수정
```python
# whole_body_env_cfg.py의 G1WholeBodyRewardsCfg에서 수정
@configclass
class CustomRewardsCfg(G1WholeBodyRewardsCfg):
    # 새로운 보상 함수 추가/수정
    my_custom_reward = RewTerm(...)
```

## 📊 환경별 비교

| 환경 | Hand | Arm | Waist | Leg | RL Dim | 용도 |
|------|------|-----|-------|-----|--------|------|
| UpperBodyIK | IK | IK | RL | RL | 15 | 보행 학습 |
| UpperBodyIL | IL | IL | RL | RL | 15 | IL+RL 결합 |
| FullRL | RL | RL | RL | RL | 43 | 전신 학습 |
| WholeBody | IK | IK | RL | RL | 15 | 기본 설정 |

## ⚠️ 주의사항

1. **Action Dimension 확인**: 환경마다 RL action dimension이 다름
2. **IL Model Loading**: IL policy 사용 시 pre-trained model 로드 필요
3. **IK Trajectory**: IK policy는 미리 정의된 trajectory 사용
4. **Performance**: Full RL은 더 많은 DOF로 인해 학습이 어려움
5. **Observation**: 모든 joint states가 observation에 포함되지만 action은 RL joints만

## 🔄 환경 사용법

Whole Body 환경은 다양한 정책 조합을 지원합니다:

```python
# Upper Body IK + Lower Body RL (가장 일반적)
env = gym.make("Isaac-Tracking-WholeBody-G1-UpperBodyIK-v0")

# Upper Body IL + Lower Body RL
env = gym.make("Isaac-Tracking-WholeBody-G1-UpperBodyIL-v0")

# Full RL Control (고급 사용자용)
env = gym.make("Isaac-Tracking-WholeBody-G1-FullRL-v0")
```
