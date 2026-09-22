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


def test_mock_intent_present(obs):
    d = MockJevClient().decide(encode_state(obs))
    assert d.meta["choices"]["INTENT"] in (
        "engage", "retreat", "circle", "advance", "attack_now")
    assert "INTENT" in d.probabilities


def test_mock_subset_questions(obs):
    d = MockJevClient().decide(encode_state(obs), questions=["ATTACK"])
    assert set(d.probabilities) == {"ATTACK"}
    assert d.meta["choices"] == {}  # INTENT/MOVEMENT_INTENT not asked


def test_scheduler_fast_decision_merges_stale(obs):
    sched = JevScheduler(MockJevClient(), cadence_hz=100.0,
                         fast_questions=("ATTACK",), strategy_period_s=1000.0)
    sched.prime(encode_state(obs))  # full bank
    full = sched.get_decision()
    fast = sched.client.decide(encode_state(obs), questions=["ATTACK"])
    merged = sched._record(fast)
    assert set(merged.probabilities) == set(QUESTION_BANK)  # constant shape
    assert merged.meta["choices"]["INTENT"] == full.meta["choices"]["INTENT"]
    assert merged.meta["questions"] == ["ATTACK"]  # honestly records the ask


def test_scheduler_salient_event_forces_strategy(obs):
    sched = JevScheduler(MockJevClient(), cadence_hz=100.0,
                         fast_questions=("ATTACK",), strategy_period_s=1000.0)
    s = encode_state(obs).model_copy(update={"enemy_visible": False})
    sched.update_state(s)
    assert not sched._force_strategy
    appeared = s.model_copy(update={"enemy_visible": True})
    sched.update_state(appeared)
    assert sched._force_strategy  # enemy appeared


def test_scheduler_episode_memory(obs):
    sched = JevScheduler(MockJevClient())
    for i in range(5):
        sched.note_episode_outcome(f"ep{i}")
    assert sched.episode_history == ["ep2", "ep3", "ep4"]  # last 3 only
    assert sched._memory()["recent_episodes"] == ["ep2", "ep3", "ep4"]
    sched.prime(encode_state(obs))
    assert sched._memory()["current_intent"] is not None
