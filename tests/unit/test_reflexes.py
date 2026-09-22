"""Aim gate (true geometry), unstuck reflex, scenario profiles."""

from __future__ import annotations

import numpy as np

from flydoom.config import apply_scenario_profile
from flydoom.doom.base import Observation
from flydoom.motor.reflexes import UnstuckReflex, aim_in_reticle


def _obs(visible=True, angle=0.0, dist=0.3, x=0.0, y=0.0):
    return Observation(frame=np.zeros((2, 2, 3), np.uint8), health=100.0,
                       ammo=10.0, kills=0, position_x=x, position_y=y,
                       angle_deg=0, enemy_visible=visible,
                       enemy_distance=dist, enemy_angle=angle, episode_tic=0)


def test_aim_in_reticle_geometry():
    assert aim_in_reticle(_obs(), 14.0, 0.45)
    assert not aim_in_reticle(_obs(visible=False), 14.0, 0.45)
    assert not aim_in_reticle(_obs(angle=0.3), 14.0, 0.45)   # 27 deg off-center
    assert not aim_in_reticle(_obs(dist=0.8), 14.0, 0.45)    # out of range
    assert aim_in_reticle(_obs(angle=0.1), 14.0, 0.45)       # 9 deg: inside


def test_unstuck_triggers_only_when_stuck_and_moving():
    u = UnstuckReflex(window_s=2.0, epsilon=5.0, turn_s=1.0)
    # commanding movement but no displacement for >2s of game time
    seen = []
    for i in range(30):  # 3s at 0.1s steps
        seen.append(u.update(100.0, 100.0, trying_to_move=True,
                             t_game_s=i * 0.1))
    fired = [d for d in seen if d]
    assert fired and set(fired) == {fired[0]}  # one trigger, one direction
    assert u.triggers == 1  # no instant re-trigger from the stale window


def test_unstuck_ignores_actual_movement():
    u = UnstuckReflex(window_s=2.0, epsilon=5.0, turn_s=1.0)
    for i in range(40):
        assert u.update(100.0 + i * 2.0, 100.0, True, i * 0.1) is None
    assert u.triggers == 0


def test_unstuck_ignores_idle():
    u = UnstuckReflex(window_s=2.0, epsilon=5.0, turn_s=1.0)
    for i in range(40):  # standing still but NOT commanding movement: fine
        assert u.update(100.0, 100.0, False, i * 0.1) is None
    assert u.triggers == 0


def test_unstuck_alternates_direction():
    u = UnstuckReflex(window_s=1.0, epsilon=5.0, turn_s=0.5)
    dirs = set()
    t = 0.0
    for _ in range(2):
        for i in range(25):  # stuck again after each forced turn ends
            t += 0.1
            d = u.update(50.0, 50.0, True, t)
            if d:
                dirs.add(d)
        t += 0.6  # let the forced turn expire
    assert dirs == {"turn_left", "turn_right"}


class _FakeFull:
    """Minimal typed connectome for TypedDNDecoder."""
    cell_type = np.array(["DNa02", "DNa02", "DNp09", "DNg02_a"])
    side = np.array([0, 1, -1, -1], dtype=np.int8)
    ids = np.array([11, 22, 33, 44])

    def type_indices(self, t):
        return np.flatnonzero(self.cell_type == t)


def _typed_decoder():
    from flydoom.motor.decoder import TypedDNDecoder
    return TypedDNDecoder(_FakeFull(), ["forward", "turn_left", "turn_right",
                                        "attack", "noop"],
                          readouts={
                              "turn": {"positive": [{"type": "DNa02", "side": "R"}],
                                       "negative": [{"type": "DNa02", "side": "L"}]},
                              "forward": {"positive": [{"type": "DNp09"}],
                                          "negative": []},
                              "attack": {"positive": [{"type": "DNg02_a"}],
                                         "negative": []}},
                          attack_gain=4.0, attack_threshold=5.0)


def test_geometry_gate_blocks_and_allows_attack():
    dec = _typed_decoder()
    rates = np.array([0.0, 0.0, 0.0, 50.0], dtype=np.float32)  # attack DN hot
    gated = dec.decode(rates, aim_ok=False)
    assert gated["scores"]["attack"] == 0.0
    assert gated["selected"] != "attack"
    assert gated["channels"]["attack"] > 0.0  # raw channel still reported
    open_ = dec.decode(rates, aim_ok=True)
    assert open_["selected"] == "attack"


def test_scenario_profile_merge():
    cfg = {"motor": {"forward_min": 0.5, "attack_aim_cone_deg": 14.0},
           "scenario_profiles": {
               "e1m1": {"motor": {"forward_min": 0.2},
                        "unstuck": {"enabled": True}}}}
    merged = apply_scenario_profile(cfg, "e1m1")
    assert merged["motor"]["forward_min"] == 0.2
    assert merged["motor"]["attack_aim_cone_deg"] == 14.0  # base preserved
    assert merged["unstuck"]["enabled"] is True
    assert apply_scenario_profile(cfg, "fly_arena") is cfg  # no profile: unchanged
    assert cfg["motor"]["forward_min"] == 0.5               # base not mutated


def test_unstuck_escape_confirmation():
    u = UnstuckReflex(window_s=2.0, epsilon=5.0, turn_s=1.0)
    t = 0.0
    for i in range(30):  # park -> trigger
        t = i * 0.1
        u.update(100.0, 100.0, True, t)
    assert u.triggers == 1
    assert u.pop_escape() is False  # no movement yet
    # fly now actually moves beyond epsilon within the escape deadline
    escaped = False
    for i in range(30, 45):
        t = i * 0.1
        u.update(100.0 + (i - 29) * 2.0, 100.0, True, t)
        escaped = escaped or u.pop_escape()
    assert escaped and u.escapes == 1
    assert u.pop_escape() is False  # one-shot


def test_unstuck_escape_deadline_expires():
    u = UnstuckReflex(window_s=2.0, epsilon=5.0, turn_s=1.0)
    for i in range(30):
        u.update(100.0, 100.0, True, i * 0.1)
    assert u.triggers == 1
    # idle (not commanding movement) past the escape deadline: no credit,
    # and no re-trigger to refresh the escape window
    for i in range(30, 90):
        u.update(100.0, 100.0, False, i * 0.1)
    assert u.escapes == 0
    # movement AFTER the deadline earns no credit
    u.update(120.0, 100.0, False, 9.0)
    assert u.pop_escape() is False and u.escapes == 0
