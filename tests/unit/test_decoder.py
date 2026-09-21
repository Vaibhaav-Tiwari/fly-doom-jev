import numpy as np

from flydoom.motor import MotorDecoder


def test_decoder_selects_most_active_bank(fixture_connectome):
    actions = ["forward", "turn_left", "turn_right", "attack", "noop"]
    dec = MotorDecoder(fixture_connectome, actions, threshold=0.01)
    rates = np.zeros(fixture_connectome.population_sizes()["descending_neuron"],
                     dtype=np.float32)
    bank_pos = dec._bank_positions("attack")
    rates[bank_pos] = 50.0
    out = dec.decode(rates)
    assert out["selected"] == "attack"
    assert out["confidence"] > 0.9
    assert out["scores"]["attack"] == max(out["scores"].values())


def test_decoder_noop_below_threshold(fixture_connectome):
    actions = ["forward", "turn_left", "turn_right", "attack", "noop"]
    dec = MotorDecoder(fixture_connectome, actions, threshold=1.0)
    out = dec.decode(np.full(48, 0.1, dtype=np.float32))  # uniform: no contrast
    assert out["selected"] == "noop"
    assert out["scores"]["noop"] == 1.0


def test_decoder_exposes_contributing_neurons(fixture_connectome):
    actions = ["forward", "turn_left", "turn_right", "attack", "noop"]
    dec = MotorDecoder(fixture_connectome, actions)
    banks = [set(dec.contributing_neurons(a)) for a in actions if a != "noop"]
    assert all(banks)
    total = set().union(*banks)
    assert len(total) == sum(len(b) for b in banks)  # disjoint banks
