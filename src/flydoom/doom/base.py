from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# scenario -> (vizdoom buttons, exposed action names). noop maps to all-zero.
SCENARIOS = {
    "basic": (["MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT", "ATTACK"],
              ["forward", "turn_left", "turn_right", "attack", "noop"]),
    "defend_the_center": (["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
                          ["turn_left", "turn_right", "attack", "noop"]),
    "defend_the_line": (["TURN_LEFT", "TURN_RIGHT", "ATTACK"],
                        ["turn_left", "turn_right", "attack", "noop"]),
    # fly_arena: custom scenario derived from defend_the_line (see
    # flydoom/scenarios/make_fly_arena.py + PROVENANCE.md). WAD ships in the
    # flydoom.scenarios package, not the vizdoom package dir.
    "fly_arena": (["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT",
                   "ATTACK"],
                  ["forward", "backward", "turn_left", "turn_right", "attack",
                   "noop"]),
    # e1m1: FreeDoom phase-1 E1M1 (freedoom1.wad bundled with ViZDoom; BSD-3 —
    # the FreeDoom replacement for DOOM E1M1; id's doom.wad is proprietary and
    # NOT shipped). Full movement, honest ammo (pistol, AMMO1, no infinite
    # ammo), skill from config.
    "e1m1": (["MOVE_FORWARD", "MOVE_BACKWARD", "TURN_LEFT", "TURN_RIGHT",
              "ATTACK"],
             ["forward", "backward", "turn_left", "turn_right", "attack",
              "noop"]),
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
