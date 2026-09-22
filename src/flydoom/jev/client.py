"""Jev clients: deterministic mock (default) and the live System One client.

Both implement the same ``JevClient`` interface and return typed JevDecision
objects. The mock is a deterministic function of the environment state — same
state in, same probabilities out — so runs are reproducible without an API key.

The live client talks to TypeSafe's System One structured-probability API
(``POST {JEV_BASE_URL}/systemone``, model ``jev-latest``): one request carries
the structured EnvironmentState as ``state`` plus the typed question bank
(noul/score/choice). The API key is read from JEV_API_KEY and NEVER leaves this
process; it is never written to recordings or logs.
"""

from __future__ import annotations

import itertools
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from flydoom.state.encoder import EnvironmentState

from .questions import QUESTION_BANK, SCORE_MAX_LEVEL, SYSTEMONE_QUESTIONS

DEFAULT_BASE_URL = "https://api.typesafe.ai/v1"
DEFAULT_MODEL = "jev-latest"
DEFAULT_DECIDE_PATH = "/systemone"
# hosted proxy variant (https://jevtypesafeai.com/api/v1) uses "/decide"


class JevError(RuntimeError):
    """Base class for live Jev failures."""


class JevAuthError(JevError):
    """Permanent failure: 401 bad key / 402 insufficient credits / 403 inactive.
    The scheduler disables live Jev for the rest of the run on this."""


class JevTransientError(JevError):
    """Transient failure (e.g. 502 upstream model error after retries)."""


@dataclass
class JevDecision:
    request_id: str
    timestamp: float
    probabilities: dict[str, float]      # question -> [0,1]
    latency_ms: float
    model: str
    is_mock: bool
    state_episode_tic: int
    meta: dict = field(default_factory=dict)


class JevClient(ABC):
    @abstractmethod
    def decide(self, state: EnvironmentState) -> JevDecision: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class MockJevClient(JevClient):
    """Deterministic heuristic stand-in for Jev. Not a neural/LLM model."""

    def __init__(self, simulated_latency_ms: float = 5.0):
        self.simulated_latency_ms = simulated_latency_ms
        self._counter = itertools.count(1)

    @property
    def name(self) -> str:
        return "mock-jev-v1"

    def decide(self, state: EnvironmentState) -> JevDecision:
        t0 = time.perf_counter()
        threat = state.threat_level
        ammo_ok = state.ammo > 0
        probs = {
            "ATTACK": _clamp(1.2 * threat - 0.15) if (state.enemy_visible and ammo_ok) else 0.0,
            "RETREAT": _clamp(0.9 * threat + (0.4 if not ammo_ok else 0.0)
                              + (0.3 if state.health < 30 else 0.0) - 0.35),
            "EXPLORE": 0.0 if state.enemy_visible else 0.8,
            "REPOSITION": _clamp(abs(state.enemy_angle) * 1.5) if state.enemy_visible else 0.1,
            "SEEK_AMMO": 1.0 if state.ammo <= 0 else (0.6 if state.ammo < 5 else 0.0),
            "THREAT_LEVEL": threat,
            "ENEMY_PRESENT": 0.95 if state.enemy_visible else 0.05,
            "MOVEMENT_INTENT": 0.7 if state.enemy_visible else 0.5,
            "TARGET_PRIORITY": 0.8 if state.enemy_visible else 0.1,
            "ENGAGEMENT_CONFIDENCE": _clamp(0.5 + 0.5 * (state.health / 100.0)
                                            - (0.4 if not ammo_ok else 0.0)),
        }
        latency = (time.perf_counter() - t0) * 1000.0 + self.simulated_latency_ms
        # deterministic choice distribution for MOVEMENT_INTENT: steer toward
        # the enemy bearing (ViZDoom convention: enemy_angle > 0 -> turn left)
        ang = float(state.enemy_angle)
        if state.enemy_visible:
            tl, tr = _clamp(ang), _clamp(-ang)
            fwd = 0.3 * (1.0 - abs(ang))
        else:
            tl, tr, fwd = 0.1, 0.1, 0.8
        total = tl + tr + fwd + 0.05
        move_dist = {"turn_left": round(tl / total, 4),
                     "turn_right": round(tr / total, 4),
                     "forward": round(fwd / total, 4),
                     "hold": round(0.05 / total, 4),
                     "strafe_left": 0.0, "strafe_right": 0.0}
        probs["MOVEMENT_INTENT"] = max(move_dist.values())
        return JevDecision(
            request_id=f"mock-{next(self._counter):06d}",
            timestamp=time.time(),
            probabilities=probs,
            latency_ms=latency,
            model=self.name,
            is_mock=True,
            state_episode_tic=state.episode_tic,
            meta={"questions": list(QUESTION_BANK),
                  "choices": {"MOVEMENT_INTENT": max(move_dist, key=move_dist.get)},
                  "choice_probabilities": {"MOVEMENT_INTENT": move_dist}},
        )


class LiveJevClient(JevClient):
    """TypeSafe System One client (structured probabilities, not chat).

    One POST to ``{base_url}/systemone`` per decision: the structured
    EnvironmentState as ``state`` and the typed question bank as ``questions``.
    Answers are parsed per question type:

    - noul   -> ``answer.noul`` (probability of yes, already 0..1)
    - score  -> ``answer.score / max_level`` (expected rubric level normalized)
    - choice -> probability of the winning choice (from ``answer.probabilities``)

    Raises at construction if JEV_API_KEY is unset so callers fail loudly
    rather than silently mocking. 502 responses are retried with backoff;
    401/402/403 raise JevAuthError (permanent — the scheduler disables live
    Jev for the rest of the run); malformed responses raise ValueError.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 timeout_s: float = 30.0, decide_path: str | None = None,
                 max_retries: int = 2):
        self.api_key = os.environ.get("JEV_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "LiveJevClient requires JEV_API_KEY in the environment "
                "(server-side only). Use jev.mode: mock otherwise.")
        self.base_url = (base_url or os.environ.get("JEV_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("JEV_MODEL") or DEFAULT_MODEL
        self.decide_path = (decide_path or os.environ.get("JEV_DECIDE_PATH")
                            or DEFAULT_DECIDE_PATH)
        self.timeout_s = timeout_s
        self.max_retries = max(0, int(max_retries))  # retries on 502 only

    @property
    def name(self) -> str:
        return f"live:{self.model}"

    def probe(self) -> dict:
        """GET /models — verifies endpoint + key without spending a decision.

        Raises on any failure (bad key, unreachable host, non-200)."""
        import httpx
        resp = httpx.get(f"{self.base_url}/models",
                         headers={"Authorization": f"Bearer {self.api_key}"},
                         timeout=10.0)
        resp.raise_for_status()
        return resp.json()

    def decide(self, state: EnvironmentState) -> JevDecision:
        import httpx

        t0 = time.perf_counter()
        body = self._post_with_retries(httpx, state)
        probs, confidence, choices, choice_probs = self._parse_answers(body)
        return JevDecision(
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
            probabilities=probs,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            model=str(body.get("model", self.model)),
            is_mock=False,
            state_episode_tic=state.episode_tic,
            meta={"questions": list(QUESTION_BANK),
                  "usage": body.get("usage", {}),
                  "confidence": confidence,
                  "choices": choices,
                  "choice_probabilities": choice_probs},
        )

    def _post_with_retries(self, httpx, state: EnvironmentState) -> dict:
        payload = {"model": self.model,
                   "state": state.model_dump(),
                   "questions": SYSTEMONE_QUESTIONS}
        for attempt in range(self.max_retries + 1):
            resp = httpx.post(
                f"{self.base_url}{self.decide_path}",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload, timeout=self.timeout_s)
            if resp.status_code == 502 and attempt < self.max_retries:
                time.sleep(2.0 ** attempt)  # 1s, 2s, ... backoff
                continue
            if resp.status_code in (401, 402, 403):
                raise JevAuthError(
                    f"Jev API {resp.status_code} ({_error_code(resp)}): "
                    "disabling live Jev for this run")
            if resp.status_code == 502:
                raise JevTransientError(
                    f"Jev API 502 upstream model error after {attempt + 1} attempts")
            resp.raise_for_status()
            return resp.json()
        raise JevTransientError("unreachable")  # pragma: no cover

    @staticmethod
    def _parse_answers(body: dict) -> tuple[dict[str, float], dict, dict, dict]:
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("System One response has no 'answers' object")
        probs: dict[str, float] = {}
        confidence: dict[str, float] = {}
        choices: dict[str, str] = {}
        choice_probs: dict[str, dict[str, float]] = {}
        for name, spec in SYSTEMONE_QUESTIONS.items():
            ans = answers.get(name)
            if not isinstance(ans, dict):
                raise ValueError(f"System One response is missing answer {name!r}")
            qtype = spec["type"]
            if ans.get("type") != qtype:
                raise ValueError(
                    f"answer {name!r}: expected type {qtype!r}, got {ans.get('type')!r}")
            if qtype == "noul":
                probs[name] = _clamp(float(ans["noul"]))
            elif qtype == "score":
                probs[name] = _clamp(float(ans["score"]) / max(SCORE_MAX_LEVEL[name], 1))
                confidence[name] = _clamp(float(ans.get("confidence", 0.0)))
            elif qtype == "choice":
                choice = str(ans["choice"])
                dist = {str(k): _clamp(float(v))
                        for k, v in (ans.get("probabilities") or {}).items()}
                p = dist.get(choice, ans.get("confidence", 0.0))
                probs[name] = _clamp(float(p))
                confidence[name] = _clamp(float(ans.get("confidence", 0.0)))
                choices[name] = choice
                choice_probs[name] = dist
            else:  # pragma: no cover - guarded by SYSTEMONE_QUESTIONS
                raise ValueError(f"unknown question type {qtype!r}")
        return probs, confidence, choices, choice_probs


def _error_code(resp) -> str:
    try:
        detail = resp.json()
        if isinstance(detail, dict):
            return str(detail.get("code") or detail.get("detail") or "")[:80]
    except Exception:
        pass
    return ""


def make_jev_client(cfg: dict) -> JevClient:
    jev = cfg["jev"]
    if jev.get("mode", "mock") == "live":
        return LiveJevClient(base_url=jev.get("base_url"), model=jev.get("model"),
                             timeout_s=float(jev.get("timeout_seconds", 30.0)),
                             decide_path=jev.get("decide_path"),
                             max_retries=int(jev.get("max_retries", 2)))
    return MockJevClient()


def _clamp(x: float) -> float:
    return float(min(1.0, max(0.0, round(x, 4))))
