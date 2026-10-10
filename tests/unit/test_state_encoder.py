import json

from flydoom.state import SCHEMA_VERSION, encode_state
from flydoom.state.encoder import compute_threat


def test_state_schema_typed_and_versioned(obs):
    s = encode_state(obs)
    assert s.schema_version == SCHEMA_VERSION
    assert s.health == 82.0 and s.ammo == 17.0 and s.kills == 1
    assert s.enemy_visible is True
    assert 0.0 <= s.enemy_distance <= 1.0
    assert -1.0 <= s.enemy_angle <= 1.0
    assert 0.0 <= s.threat_level <= 1.0
    assert "attack" in s.available_actions


def test_threat_heuristic_bounds(obs):
    assert compute_threat(obs) > 0.5  # close, near-centered enemy
    obs.enemy_visible = False
    assert compute_threat(obs) == 0.0
    obs.enemy_visible = True
    obs.enemy_distance = 1.0
    obs.enemy_angle = 1.0
    assert compute_threat(obs) <= 0.05


def test_state_serializes(obs):
    s = encode_state(obs)
    assert "threat_level" in s.model_dump_json()


def test_goal_can_be_attached_and_serialized_without_schema_change(obs):
    state = encode_state(obs)
    goal = {
        "exit_dist_units": 256.5,
        "exit_bearing_deg": -45.0,
        "note": "Reach the E1M1 exit lift.",
    }

    state.goal = goal
    serialized = json.loads(state.model_dump_json())

    assert serialized["schema_version"] == SCHEMA_VERSION
    assert serialized["goal"] == goal


def test_goal_is_omitted_for_ordinary_arena_observations(obs):
    state = encode_state(obs)

    assert state.goal is None
    assert "goal" not in state.model_dump(exclude_none=True)
    assert state.schema_version == SCHEMA_VERSION
