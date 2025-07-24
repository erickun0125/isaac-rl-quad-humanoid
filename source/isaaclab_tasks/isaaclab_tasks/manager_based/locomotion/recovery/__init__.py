# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Locomotion environments with recovery tasks.

These environments focus on teaching the robot to recover from fallen or unstable states.
The robot learns to get back to a stable walking position from various initial poses.
"""

import gymnasium as gym

from . import recovery_env_cfg

##
# Register Gym environments.
##

##
# Recovery environments.
##