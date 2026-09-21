import numpy as np
import pytest

from flydoom.integration import JevBridge, MotorInjectionError
from flydoom.jev.client import MockJevClient
from flydoom.state import encode_state


def test_motor_injection_rejected(fixture_connectome):
    with pytest.raises(MotorInjectionError):
        JevBridge(fixture_connectome, {"ATTACK": "descending_neuron"})


def test_unknown_population_rejected(fixture_connectome):
    with pytest.raises(ValueError):
        JevBridge(fixture_connectome, {"ATTACK": "not_a_population"})


def test_modulation_targets_only_allowed_populations(fixture_connectome, obs):
    bridge = JevBridge(fixture_connectome, {"ATTACK": "visual_projection",
                                            "EXPLORE": "cx_intrinsic"}, gain=10.0)
    decision = MockJevClient().decide(encode_state(obs))
    idx, cur = bridge.modulation_currents(decision)
    assert len(idx) == len(cur) and len(idx) > 0
    pops = fixture_connectome.population[idx]
    from flydoom.malecns.graph import MOTOR_POPULATION, POPULATIONS
    assert MOTOR_POPULATION not in {POPULATIONS[p] for p in set(pops.tolist())}
    # ATTACK probability > 0 for this state -> some nonzero current
    assert np.any(cur > 0)


def test_modulation_none_decision_is_all_zero(fixture_connectome):
    bridge = JevBridge(fixture_connectome, {"ATTACK": "visual_projection"})
    idx, cur = bridge.modulation_currents(None)
    assert len(idx) > 0 and np.all(cur == 0.0)
