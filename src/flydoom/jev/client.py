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
        return JevDecision(
            request_id=f"mock-{next(self._counter):06d}",
            timestamp=time.time(),
            probabilities=probs,
            latency_ms=latency,
            model=self.name,
            is_mock=True,
            state_episode_tic=state.episode_tic,
            meta={"questions": list(QUESTION_BANK)},
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
    rather than silently mocking. Raises on HTTP errors, malformed responses,
    or missing/mistyped answers — the scheduler keeps the last valid decision.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 timeout_s: float = 30.0):
        self.api_key = os.environ.get("JEV_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "LiveJevClient requires JEV_API_KEY in the environment "
                "(server-side only). Use jev.mode: mock otherwise.")
        self.base_url = (base_url or os.environ.get("JEV_BASE_URL")
                         or DEFAULT_BASE_URL).rstrip("/")
        self.model = model or os.environ.get("JEV_MODEL") or DEFAULT_MODEL
        self.timeout_s = timeout_s

    @property
    def name(self) -> str:
        return f"live:{self.model}"

    def decide(self, state: EnvironmentState) -> JevDecision:
        import httpx

        t0 = time.perf_counter()
        resp = httpx.post(
            f"{self.base_url}/systemone",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model,
                  "state": state.model_dump(),
                  "questions": SYSTEMONE_QUESTIONS},
            timeout=self.timeout_s)
        resp.raise_for_status()
        body = resp.json()
        probs, confidence, choices = self._parse_answers(body)
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
                  "choices": choices},
        )

    @staticmethod
    def _parse_answers(body: dict) -> tuple[dict[str, float], dict, dict]:
        answers = body.get("answers")
        if not isinstance(answers, dict):
            raise ValueError("System One response has no 'answers' object")
        probs: dict[str, float] = {}
        confidence: dict[str, float] = {}
        choices: dict[str, str] = {}
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
                dist = ans.get("probabilities") or {}
                p = dist.get(choice, ans.get("confidence", 0.0))
                probs[name] = _clamp(float(p))
                confidence[name] = _clamp(float(ans.get("confidence", 0.0)))
                choices[name] = choice
            else:  # pragma: no cover - guarded by SYSTEMONE_QUESTIONS
                raise ValueError(f"unknown question type {qtype!r}")
        return probs, confidence, choices


def make_jev_client(cfg: dict) -> JevClient:
    jev = cfg["jev"]
    if jev.get("mode", "mock") == "live":
        return LiveJevClient(base_url=jev.get("base_url"), model=jev.get("model"),
                             timeout_s=float(jev.get("timeout_seconds", 30.0)))
    return MockJevClient()


def _clamp(x: float) -> float:
    return float(min(1.0, max(0.0, round(x, 4))))
