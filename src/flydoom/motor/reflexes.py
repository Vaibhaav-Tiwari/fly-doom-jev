"""Control reflexes against TRUE game geometry (not neural proxies).

- `aim_in_reticle`: the working attack aim gate. Uses the observation's
  ViZDoom-label-derived enemy fields (visible, aim offset, distance) — real
  enemy geometry, replacing the old neural turn-imbalance gate that starved
  the attack channel. Attack fires only when an enemy is inside the aim cone
  of screen center, visible, and within range.
- `UnstuckReflex`: wall-stuck detection. If the agent commands movement but
  its XY position barely changes over `window_s` of game time, force a turn
  for `turn_s` (alternating direction each trigger). Classic unstuck reflex;
  chosen engineering dynamics, not biology (config + SCIENCE.md).
"""

from __future__ import annotations

import math
from collections import deque


def aim_in_reticle(obs, cone_deg: float = 14.0, range_max: float = 0.45) -> bool:
    """True iff an enemy is visible, inside the aim cone, and within range.

    obs.enemy_angle is the signed label-bearing offset normalized to [-1, 1]
    per ±90°; obs.enemy_distance is 0 (point-blank) .. 1 (far/not visible).
    """
    return bool(obs.enemy_visible
                and abs(float(obs.enemy_angle)) * 90.0 <= cone_deg
                and float(obs.enemy_distance) <= range_max)


class UnstuckReflex:
    """Force a turn when commanded movement produces no displacement."""

    def __init__(self, window_s: float = 2.5, epsilon: float = 6.0,
                 turn_s: float = 1.0):
        self.window_s = float(window_s)
        self.epsilon = float(epsilon)   # map units over the window
        self.turn_s = float(turn_s)
        self._hist: deque = deque()
        self._forced_until = -1.0
        self._direction = "turn_right"  # flips on every trigger
        self.triggers = 0

    def reset(self) -> None:
        self._hist.clear()
        self._forced_until = -1.0

    def update(self, x: float, y: float, trying_to_move: bool,
               t_game_s: float) -> str | None:
        """Forced turn action while active, else None."""
        if t_game_s < self._forced_until:
            return self._direction
        self._hist.append((t_game_s, float(x), float(y)))
        while self._hist and self._hist[0][0] < t_game_s - self.window_s:
            self._hist.popleft()
        if not trying_to_move or len(self._hist) < 2:
            return None
        t0, x0, y0 = self._hist[0]
        if t_game_s - t0 < self.window_s * 0.9:  # window not yet full
            return None
        if math.hypot(x - x0, y - y0) >= self.epsilon:
            return None
        self._forced_until = t_game_s + self.turn_s
        self._hist.clear()  # fresh window: the forced turn (and whatever it
                            # achieved) must not count as continued stuckness
        self._direction = ("turn_left" if self._direction == "turn_right"
                           else "turn_right")
        self.triggers += 1
        return self._direction
