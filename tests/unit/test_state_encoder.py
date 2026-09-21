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
