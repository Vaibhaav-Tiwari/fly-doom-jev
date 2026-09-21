"""LiveJevClient tests — System One transport, fully mocked (no network, no key).

Covers noul/score/choice answer parsing, usage/confidence/choice metadata,
and loud failure on missing key, HTTP errors, and malformed answers.
"""

import httpx
import pytest

from flydoom.jev.client import (DEFAULT_BASE_URL, DEFAULT_MODEL, JevAuthError,
                                JevTransientError, LiveJevClient)
from flydoom.jev.questions import QUESTION_BANK, SYSTEMONE_QUESTIONS
from flydoom.jev.scheduler import JevScheduler
from flydoom.state import encode_state

KEY = "test-key-not-a-real-secret"


def _systemone_payload() -> dict:
    answers = {}
    for name, spec in SYSTEMONE_QUESTIONS.items():
        if spec["type"] == "noul":
            answers[name] = {"type": "noul", "noul": 0.89}
        elif spec["type"] == "score":
            answers[name] = {"type": "score", "score": 2.0, "confidence": 0.92,
                             "probabilities": {"0": 0.0, "1": 0.1, "2": 0.8, "3": 0.1}}
        else:
            choice = next(iter(spec["criteria"]))
            answers[name] = {"type": "choice", "choice": choice, "confidence": 0.7,
                             "probabilities": {choice: 0.6, "other": 0.4}}
    return {"model": "jev-1.13.0", "answers": answers,
            "usage": {"input_tokens": 539, "output_tokens": 89,
                      "cost_usd": 0.0042, "credits_remaining_usd": 9.99}}


def _mock_post(monkeypatch, payload=None, status=200, capture=None):
    def fake_post(url, headers=None, json=None, timeout=None):
        if capture is not None:
            capture.update(url=url, headers=headers, json=json, timeout=timeout)
        req = httpx.Request("POST", url)
        return httpx.Response(status, json=payload or {}, request=req)
    monkeypatch.setattr(httpx, "post", fake_post)


@pytest.fixture()
def live_client(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", KEY)
    monkeypatch.delenv("JEV_BASE_URL", raising=False)
    monkeypatch.delenv("JEV_MODEL", raising=False)
    return LiveJevClient()


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("JEV_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="JEV_API_KEY"):
        LiveJevClient()


def test_decide_parses_all_answer_types(live_client, monkeypatch, obs):
    capture = {}
    _mock_post(monkeypatch, _systemone_payload(), capture=capture)
    d = live_client.decide(encode_state(obs))

    assert capture["url"] == f"{DEFAULT_BASE_URL}/systemone"
    assert capture["headers"]["Authorization"] == f"Bearer {KEY}"
    assert capture["json"]["model"] == DEFAULT_MODEL
    assert set(capture["json"]["questions"]) == set(QUESTION_BANK)
    assert "frame" not in capture["json"]["state"]  # pixels never leave

    assert set(d.probabilities) == set(QUESTION_BANK)
    assert not d.is_mock
    assert d.model == "jev-1.13.0"  # model name from the response
    assert d.probabilities["ATTACK"] == 0.89                       # noul passthrough
    assert d.probabilities["THREAT_LEVEL"] == pytest.approx(2.0 / 3.0, abs=1e-3)  # score normalized
    assert d.probabilities["MOVEMENT_INTENT"] == 0.6               # winner probability
    assert d.meta["usage"]["input_tokens"] == 539
    assert d.meta["usage"]["cost_usd"] == 0.0042
    assert d.meta["usage"]["credits_remaining_usd"] == 9.99
    assert d.meta["confidence"]["THREAT_LEVEL"] == 0.92
    assert d.meta["choices"]["MOVEMENT_INTENT"] == "forward"
    assert d.meta["choice_probabilities"]["MOVEMENT_INTENT"] == {"forward": 0.6,
                                                                 "other": 0.4}


def test_decide_raises_on_http_error(live_client, monkeypatch, obs):
    _mock_post(monkeypatch, {"detail": "bad body"}, status=400)
    with pytest.raises(httpx.HTTPStatusError):
        live_client.decide(encode_state(obs))


def test_decide_raises_on_missing_answer(live_client, monkeypatch, obs):
    payload = _systemone_payload()
    del payload["answers"]["RETREAT"]
    _mock_post(monkeypatch, payload)
    with pytest.raises(ValueError, match="RETREAT"):
        live_client.decide(encode_state(obs))


def test_decide_raises_on_type_mismatch(live_client, monkeypatch, obs):
    payload = _systemone_payload()
    payload["answers"]["ATTACK"] = {"type": "score", "score": 1.0, "confidence": 0.5}
    _mock_post(monkeypatch, payload)
    with pytest.raises(ValueError, match="expected type"):
        live_client.decide(encode_state(obs))


class _SequencePost:
    """Fake httpx.post returning a scripted sequence of status codes."""

    def __init__(self, statuses, payload):
        self.statuses = list(statuses)
        self.payload = payload
        self.calls = 0

    def __call__(self, url, headers=None, json=None, timeout=None):
        status = self.statuses[min(self.calls, len(self.statuses) - 1)]
        self.calls += 1
        body = self.payload if status == 200 else {"detail": "err", "code": "x"}
        return httpx.Response(status, json=body,
                              request=httpx.Request("POST", url))


def test_502_retried_with_backoff(live_client, monkeypatch, obs):
    fake = _SequencePost([502, 502, 200], _systemone_payload())
    monkeypatch.setattr(httpx, "post", fake)
    monkeypatch.setattr("flydoom.jev.client.time.sleep", lambda s: None)
    d = live_client.decide(encode_state(obs))
    assert fake.calls == 3
    assert d.probabilities["ATTACK"] == 0.89


def test_502_exhausted_raises_transient(live_client, monkeypatch, obs):
    fake = _SequencePost([502], _systemone_payload())
    monkeypatch.setattr(httpx, "post", fake)
    monkeypatch.setattr("flydoom.jev.client.time.sleep", lambda s: None)
    with pytest.raises(JevTransientError):
        live_client.decide(encode_state(obs))
    assert fake.calls == 1 + live_client.max_retries


@pytest.mark.parametrize("status", [401, 402, 403])
def test_auth_errors_raise_permanent_without_retry(live_client, monkeypatch,
                                                   obs, status):
    fake = _SequencePost([status], _systemone_payload())
    monkeypatch.setattr(httpx, "post", fake)
    with pytest.raises(JevAuthError):
        live_client.decide(encode_state(obs))
    assert fake.calls == 1  # no retries on permanent failures


def test_decide_path_configurable(monkeypatch):
    monkeypatch.setenv("JEV_API_KEY", KEY)
    c = LiveJevClient(base_url="https://jevtypesafeai.com/api/v1",
                      decide_path="/decide")
    capture = {}
    _mock_post(monkeypatch, _systemone_payload(), capture=capture)
    assert c.decide_path == "/decide"


def test_scheduler_disables_on_auth_error(monkeypatch, obs):
    monkeypatch.setenv("JEV_API_KEY", KEY)
    client = LiveJevClient()
    _mock_post(monkeypatch, {"code": "insufficient_credits"}, status=402)
    sched = JevScheduler(client, cadence_hz=50.0)
    assert sched.prime(encode_state(obs)) is None
    assert sched.disabled_reason is not None
    assert "402" in sched.disabled_reason
    assert sched.errors == 1


def test_scheduler_tracks_cost(monkeypatch, obs):
    monkeypatch.setenv("JEV_API_KEY", KEY)
    client = LiveJevClient()
    _mock_post(monkeypatch, _systemone_payload())
    sched = JevScheduler(client, cadence_hz=50.0)
    sched.prime(encode_state(obs))
    sched.prime(encode_state(obs))
    assert sched.decisions_made == 2
    assert sched.total_cost_usd == pytest.approx(2 * 0.0042)
    assert sched.credits_remaining_usd == 9.99
