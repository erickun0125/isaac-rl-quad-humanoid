# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MDP terms for recovery locomotion tasks."""

# Import velocity MDP terms that we'll reuse
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa

# Import custom recovery-specific terms
from .rewards import *  # noqa
from .events import *  # noqa
from .curriculums import *  # noqa
from .observations import *  # noqa
