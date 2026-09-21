from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# scenario -> (vizdoom buttons, exposed action names). noop maps to all-zero.
SCENARIOS = {
    "basic": (["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "ATTACK"],
              ["forward", "turn_left", "turn_right", "attack", "noop"]),
    "defend_the_center": (["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
                          ["turn_left", "turn_right", "attack", "noop"]),
    "deadly_corridor": (["MOVE_LEFT", "MOVE_RIGHT", "MOVE_FORWARD",
                         "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT", "ATTACK"],
                        ["left", "right", "forward", "backward",
                         "turn_left", "turn_right", "attack", "noop"]),
}


@dataclass
class Observation:
    """One environment observation shared by both backends."""

    frame: np.ndarray            # (H, W, 3) uint8 RGB
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
