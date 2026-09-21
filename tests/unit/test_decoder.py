import numpy as np

from flydoom.motor.decoder import BankDecoder

ACTIONS = ["forward", "turn_left", "turn_right", "attack", "noop"]


def _full_rates(conn, motor_rates):
    full = np.zeros(conn.n_neurons, dtype=np.float32)
    full[conn.population_indices("descending_neuron")] = motor_rates
    return full


def test_decoder_selects_most_active_bank(fixture_connectome):
    dec = BankDecoder(fixture_connectome, ACTIONS, threshold=0.01)
    n_motor = fixture_connectome.population_sizes()["descending_neuron"]
    rates = np.zeros(n_motor, dtype=np.float32)
    bank = dec.banks["attack"]
    pos = np.searchsorted(dec.motor_indices, bank)
    rates[pos] = 50.0
    out = dec.decode(_full_rates(fixture_connectome, rates))
    assert out["selected"] == "attack"
    assert out["confidence"] > 0.9
    assert out["scores"]["attack"] == max(out["scores"].values())


def test_decoder_noop_below_threshold(fixture_connectome):
    dec = BankDecoder(fixture_connectome, ACTIONS, threshold=1.0)
    out = dec.decode(_full_rates(fixture_connectome, np.full(48, 0.1, dtype=np.float32)))
    assert out["selected"] == "noop"
    assert out["scores"]["noop"] == 1.0


def test_decoder_banks_disjoint(fixture_connectome):
    dec = BankDecoder(fixture_connectome, ACTIONS)
    banks = [set(map(int, b)) for a, b in dec.banks.items()]
    assert all(banks)
    total = set().union(*banks)
    assert len(total) == sum(len(b) for b in banks)


def test_decoder_contributing_sets(fixture_connectome):
    dec = BankDecoder(fixture_connectome, ACTIONS)
    desc = dec.describe()
    contributing = desc["contributing"]
    assert set(contributing) == {"forward", "turn_left", "turn_right", "attack"}
    for action, bank in dec.banks.items():
        entry = contributing[action]
        assert entry["indices"] == [int(i) for i in bank]
        assert len(entry["body_ids"]) == len(bank)
        # body ids match the graph's own id array
        for i, bid in zip(bank, entry["body_ids"]):
            assert bid == int(fixture_connectome.body_ids[int(i)])
