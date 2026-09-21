import time

from flydoom.jev.client import JevDecision, MockJevClient
from flydoom.jev.questions import QUESTION_BANK
from flydoom.jev.scheduler import JevScheduler
from flydoom.state import encode_state


def test_mock_jev_deterministic(obs):
    c = MockJevClient()
    s = encode_state(obs)
    a, b = c.decide(s), c.decide(s)
    assert a.probabilities == b.probabilities
    assert set(a.probabilities) == set(QUESTION_BANK)
    assert all(0.0 <= p <= 1.0 for p in a.probabilities.values())
    assert a.is_mock


def test_mock_jev_never_sees_pixels(obs):
    # the state passed to Jev contains no frame/pixel data
    s = encode_state(obs)
    assert "frame" not in s.model_dump()


class _FailingClient(MockJevClient):
    def decide(self, state):
        raise RuntimeError("simulated Jev outage")


def test_scheduler_keeps_stale_decision_on_failure(obs):
    s = encode_state(obs)
    good = JevScheduler(MockJevClient(), cadence_hz=50.0)
    good.prime(s)
    decision = good.get_decision()
    assert decision is not None

    bad = JevScheduler(_FailingClient(), cadence_hz=50.0)
    bad._latest = decision  # last valid decision stays active
    bad.start()
    bad.update_state(s)
    time.sleep(0.1)
    bad.stop()
    assert bad.errors > 0                     # failures logged, not fatal
    assert bad.get_decision() is decision     # stale decision still served


def test_scheduler_no_decision_before_first(obs):
    sched = JevScheduler(MockJevClient(), cadence_hz=50.0)
    assert sched.get_decision() is None
