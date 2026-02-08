# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**quad-humanoid-envs** — Custom GPU-accelerated RL environments for Unitree GO2 quadruped and G1 humanoid robots, built as an extension package on top of NVIDIA Isaac Lab (v2.1.1) and Isaac Sim 4.5.

This is NOT a fork of Isaac Lab. Isaac Lab is a pip dependency; this repository contains only custom environment code.

## Common Commands

### Training (RSL-RL PPO)

```bash
# GO2 velocity tracking
python scripts/train.py --task Isaac-Velocity-Sequor-Unitree-Go2-v0 --num_envs 4096

# G1 loco-manipulation
python scripts/train.py --task Isaac-LocoManip-G1-v0 --num_envs 4096

# G1 loco-manipulation with EE tracking
python scripts/train.py --task Isaac-LocoManip-EE-G1-v0 --num_envs 4096

# G1 whole-body control (IK upper body)
python scripts/train.py --task Isaac-WholeBody-G1-UpperBodyIK-v0 --num_envs 4096

# Resume training from checkpoint
python scripts/train.py --task Isaac-LocoManip-G1-v0 --resume --load_run <run_dir>
```

### Evaluation / Play

```bash
python scripts/play.py --task Isaac-LocoManip-G1-Play-v0 --load_run <run_dir>
python scripts/play.py --task Isaac-WholeBody-G1-UpperBodyIK-Play-v0 --load_run <run_dir> --video
```

### Installation

```bash
pip install -e .
```

### Linting & Formatting

```bash
black --line-length 120 <file>
isort --profile black <file>
flake8 <file>
```

## Registered Gym Environments

| Environment ID | Robot | Task |
|---|---|---|
| `Isaac-Velocity-Sequor-Unitree-Go2-v0` | GO2 | Velocity tracking (flat) |
| `Isaac-Velocity-Sequor-Rough-Unitree-Go2-v0` | GO2 | Velocity tracking (rough) |
| `Isaac-Recovery-Sequor-Flat-Unitree-Go2-v0` | GO2 | Fall recovery (flat) |
| `Isaac-Recovery-Sequor-Rough-Unitree-Go2-v0` | GO2 | Fall recovery (rough) |
| `Isaac-LocoManip-G1-v0` | G1 | Loco-manipulation |
| `Isaac-LocoManip-EE-G1-v0` | G1 | Loco-manip + EE tracking |
| `Isaac-WholeBody-G1-UpperBodyIK-v0` | G1 | Whole-body (RL legs + IK arms) |
| `Isaac-WholeBody-G1-UpperBodyIL-v0` | G1 | Whole-body (RL legs + IL arms) |
| `Isaac-WholeBody-G1-FullRL-v0` | G1 | Whole-body (full RL) |

All environments also have `-Play-v0` variants. GO2 velocity also has `-RNN-` variants.

## Package Structure

```
quad_humanoid_envs/
├── __init__.py                     # Package root (triggers gym registration)
├── go2/
│   ├── __init__.py                 # GO2 gym registrations
│   ├── velocity/                   # Velocity tracking (flat + rough)
│   │   ├── env_cfg.py              # ManagerBasedRLEnvCfg subclasses
│   │   ├── mdp.py                  # Custom rewards, curriculums, events
│   │   └── agents/{ppo_cfg.py, symmetry.py}
│   └── recovery/                   # Fall recovery
│       ├── env_cfg.py
│       ├── mdp.py                  # Multi-phase recovery rewards
│       └── agents/{ppo_cfg.py, symmetry.py}
├── g1/
│   ├── locomanip/                  # Locomotion + manipulation
│   │   ├── env_cfg.py              # Base config
│   │   ├── env_cfg_ee.py           # EE tracking variant (inherits base)
│   │   ├── mdp.py                  # Velocity/reward curricula, foot clearance
│   │   ├── robots.py               # G1 29-DOF articulation config
│   │   └── agents/{ppo_cfg.py, symmetry.py, symmetry_ee.py}
│   └── wholebody/                  # Whole-body control
│       ├── env_cfg.py              # UpperBodyIK/IL/FullRL variants
│       ├── actions.py              # Hybrid action space (PolicyType: RL/IK/IL)
│       ├── robots.py               # G1 floating + fixed base configs
│       ├── agents/{ppo_cfg.py, symmetry.py}
│       └── controllers/{ik_controller.py, il_controller.py}
scripts/
├── train.py                        # Training entry point
└── play.py                         # Evaluation entry point
```

## Architecture

### Manager-Based Environment Pattern

Environments are composed via **managers** configured through `@configclass` dataclasses:

- **Scene**: Robot articulation + terrain + objects
- **ObservationManager**: Assembles observation vector from sensors/state
- **ActionManager**: Maps policy output → motor commands
- **RewardManager**: Individual weighted reward terms (combined additively)
- **TerminationManager**: Episode reset conditions
- **CurriculumManager**: Progressive difficulty scheduling
- **EventManager**: Domain randomization on resets

### Key Patterns

- **Config-as-Code**: All configs use `@configclass` dataclasses, not YAML
- **Symmetry Augmentation**: L-R mirror for locomotion (doubles training data)
- **Two-Level Curriculum**: Velocity range + reward weight scheduling
- **Hybrid Control** (wholebody): RL lower body + IK/IL upper body via `PolicyType` enum

## Code Style

- **Line length**: 120
- **Formatter**: Black
- **Import sorting**: isort (black profile)
- **Python**: 3.10+
- **Docstrings**: Google convention
