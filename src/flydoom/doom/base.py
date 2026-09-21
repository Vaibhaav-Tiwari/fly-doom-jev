from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ACTIONS = ["forward", "turn_left", "turn_right", "attack", "noop"]


@dataclass
class Observation:
    """One environment observation shared by both backends."""

    frame: np.ndarray            # (H, W) float32 in [0, 1]
    health: float
    ammo: float
    kills: int
    position_x: float
    position_y: float
    angle_deg: float             # player heading, degrees
    enemy_visible: bool
    enemy_distance: float        # normalized [0,1]; 1.0 when not visible
    enemy_angle: float           # signed offset from heading, normalized to [-1,1]
    episode_tic: int


@dataclass
class StepResult:
    observation: Observation
    reward: float
    done: bool
    info: dict = field(default_factory=dict)
