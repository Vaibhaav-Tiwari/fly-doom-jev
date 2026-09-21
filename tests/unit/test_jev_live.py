"""LiveJevClient tests — System One transport, fully mocked (no network, no key).

Covers noul/score/choice answer parsing, usage/confidence/choice metadata,
and loud failure on missing key, HTTP errors, and malformed answers.
"""

import httpx
import pytest

from flydoom.jev.client import DEFAULT_BASE_URL, DEFAULT_MODEL, LiveJevClient
from flydoom.jev.questions import QUESTION_BANK, SYSTEMONE_QUESTIONS
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
            "usage": {"input_tokens": 539, "output_tokens": 89}}


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
    assert d.meta["confidence"]["THREAT_LEVEL"] == 0.92
    assert d.meta["choices"]["MOVEMENT_INTENT"] == "advance"


def test_decide_raises_on_http_error(live_client, monkeypatch, obs):
    _mock_post(monkeypatch, {"detail": "unauthorized"}, status=401)
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
