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


def test_turn_offset_zeroes_standing_bias():
    import numpy as np
    from flydoom.motor.decoder import TypedDNDecoder
    from tests.unit.test_reflexes import _FakeFull, _typed_decoder
    dec = _typed_decoder()
    rates = np.array([5.0, 20.0, 0.0, 0.0], dtype=np.float32)  # idx1 = DNa02 R hot
    imb = dec.turn_imbalance(rates)
    assert imb > 0  # rightward
    before = dec.decode(rates)
    assert before["scores"]["turn_right"] > 0
    dec.set_turn_offset(imb)  # spawn-calibrated zero
    after = dec.decode(rates)
    assert after["scores"]["turn_right"] == 0.0
    assert after["scores"]["turn_left"] == 0.0


def test_settle_mutes_motor_and_calibrates():
    import numpy as np
    from flydoom.neural.settle import settle_episode_start
    from tests.unit.test_reflexes import _typed_decoder

    class FakeEngine:
        def __init__(self):
            self.rate = np.zeros(4, dtype=np.float32)
        def inject_input(self, i, c):
            pass
        def step(self, n):
            self.rate[:] = [5.0, 20.0, 0.0, 0.0]  # settles to a right-biased readout

    class FakeVision:
        def sample(self, frame, ms):
            return np.array([0]), np.array([1.0], dtype=np.float32)

    dec = _typed_decoder()
    eng = FakeEngine()
    meta = settle_episode_start(eng, FakeVision(), "photoreceptor",
                                np.zeros((2, 2, 3), np.uint8), dec, None,
                                steps_per_iter=10, dt_ms=1.0, settle_ms=40.0)
    assert meta["enabled"] and meta["iterations"] == 4 and meta["motor_muted"]
    assert dec.turn_offset != 0.0
    assert meta["turn_offset"] == round(dec.turn_offset, 4)
    # and a disabled settle is a no-op
    meta0 = settle_episode_start(eng, FakeVision(), "photoreceptor",
                                 np.zeros((2, 2, 3), np.uint8), dec, None,
                                 10, 1.0, 0.0)
    assert meta0 == {"enabled": False}
