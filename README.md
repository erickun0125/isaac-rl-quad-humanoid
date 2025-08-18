# Isaac Lab RL Environments for Quadruped and Humanoid Robots

[![IsaacSim](https://img.shields.io/badge/IsaacSim-4.5.0-silver.svg)](https://docs.isaacsim.omniverse.nvidia.com/latest/index.html)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![License](https://img.shields.io/badge/license-BSD--3-clause-yellow.svg)](https://opensource.org/licenses/BSD-3-Clause)

This repository contains Isaac Lab-based reinforcement learning policy implementations for both **quadruped** and **humanoid** robots. 

## Repository Structure

- **`feat/go2` branch** (current): Unitree Go2 quadruped locomotion environments
- **`feat/g1` branch**: Unitree G1 humanoid loco-manipulation environments

## Current Branch: Unitree Go2 Locomotion (feat/go2)

This branch presents a implementation of two critical locomotion tasks for the Unitree Go2 quadruped robot using the NVIDIA Isaac Lab framework. The work was developed during a **Internship at Sequor Robotics** as demonstration RL locomotion policies for **Sim2Real Pipeline System**.

### Important Note
The RL policy training environments and methodologies presented here represent the **publicly shareable portion** of the internship work. The proprietary **Sim2Real pipeline and deployment infrastructure remain confidential** and are not included in this repository, as they constitute Sequor Robotics' intellectual property.

Our work demonstrates successful sim-to-real transfer, achieving robust performance on the physical Unitree Go2 Edu robot through Sequor Sim2Real Pipeline.

## Abstract

We implement and evaluate two fundamental locomotion behaviors: **Recovery** (getting up from fallen states) and **Velocity Tracking** (following commanded velocities). Our approach leverages manager based environemnt in IsaacLab and ppo runner in RSLRL.

## Demo Videos

### Recovery Task
<table>
<tr>
<td width="50%" align="center">
<h4>Simulation</h4>
<video width="100%" controls preload="metadata">
  <source src="./docs/source/go2/recovery_sim.webm" type="video/webm">
  <source src="./docs/source/go2/recovery_real.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
</td>
<td width="50%" align="center">
<h4>Real Robot</h4>
<video width="100%" controls preload="metadata">
  <source src="./docs/source/go2/recovery_real.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
</td>
</tr>
</table>

### Velocity Tracking Task
<table>
<tr>
<td width="50%" align="center">
<h4>Simulation</h4>
<video width="100%" controls preload="metadata">
  <source src="./docs/source/go2/velocity_sim.webm" type="video/webm">
  <source src="./docs/source/go2/velocity_real.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
</td>
<td width="50%" align="center">
<h4>Real Robot</h4>
<video width="100%" controls preload="metadata">
  <source src="./docs/source/go2/velocity_real.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
</td>
</tr>
</table>

## Results

### Simulation Performance
- **Recovery Task**: 95%+ success rate within 5 seconds
- **Velocity Task**: <0.1 m/s average tracking error

### Sim-to-Real Transfer
- **Performance Retention**: >90% of simulation performance maintained
- **Robustness**: Successful deployment across various indoor/outdoor environments
- **Hardware Validation**: Tested on Unitree Go2 Edu platform

## Usage

### Prerequisites
- NVIDIA Isaac Sim 4.5+
- Python 3.10+
- CUDA 11.8+
- Isaac Lab framework


### Training
```bash
# Recovery task
python scripts/rsl_rl/train.py --task Isaac-Recovery-Sequor-Flat-Unitree-Go2-v0 --num_envs 4096

# Velocity task
python scripts/rsl_rl/train.py --task Isaac-Velocity-Sequor-Flat-Unitree-Go2-v0 --num_envs 4096
```

### Evaluation
```bash
# Recovery evaluation
python scripts/rsl_rl/play.py --task Isaac-Recovery-Sequor-Flat-Unitree-Go2-Play-v0 --num_envs 50

# Velocity evaluation
python scripts/rsl_rl/play.py --task Isaac-Velocity-Sequor-Flat-Unitree-Go2-Play-v0 --num_envs 50
```

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{isaac_rl_quad_humanoid,
  title={Isaac Lab RL Environments for Quadruped and Humanoid Robots},
  author={[Kyungseo Park]},
  year={2025},
  note={feat/go2 branch: Unitree Go2 Locomotion Environments}
}
```

## License

This project is licensed under the BSD-3-Clause License - see the [LICENSE](LICENSE) file for details.

---

**Contact**: [erickun0125@snu.ac.kr]   
**Project Duration**: 2025    
**Base Repository**: IsaacLab    
**Hardware Platform**: Unitree Go2 Edu