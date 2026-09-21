"""Jev clients: deterministic mock (default) and a thin live-API stub.

Both implement the same ``JevClient`` interface and return typed JevDecision
objects. The mock is a deterministic function of the environment state — same
state in, same probabilities out — so runs are reproducible without an API key.

The live client is intentionally thin in v1 (no retries/caching/cost tracking);
the API key is read from JEV_API_KEY and NEVER leaves this process.
"""

from __future__ import annotations

import itertools
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from flydoom.state.encoder import EnvironmentState

from .questions import QUESTION_BANK


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
            "THREAT_LEVEL": threat,
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
    """Thin OpenAI-compatible chat client (Moonshot/Kimi). v1 stub.

    No retry/cache/cost tracking yet (deferred). Raises at construction if
    JEV_API_KEY is unset so callers fail loudly rather than silently mocking.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None,
                 timeout_s: float = 10.0):
        self.api_key = os.environ.get("JEV_API_KEY", "")
        if not self.api_key:
            raise RuntimeError(
                "LiveJevClient requires JEV_API_KEY in the environment "
                "(server-side only). Use jev.mode: mock otherwise.")
        self.base_url = (base_url or os.environ.get("JEV_BASE_URL")
                         or "https://api.moonshot.ai/v1").rstrip("/")
        self.model = model or os.environ.get("JEV_MODEL") or "moonshot-v1-8k"
        self.timeout_s = timeout_s

    @property
    def name(self) -> str:
        return f"live:{self.model}"

    def decide(self, state: EnvironmentState) -> JevDecision:
        import httpx

        prompt = (
            "You are a probabilistic decision module. Given this game state JSON, "
            "respond with STRICT JSON mapping each question to a probability in "
            "[0,1]: " + ", ".join(QUESTION_BANK) + ".\nState: "
            + state.model_dump_json())
        t0 = time.perf_counter()
        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={"model": self.model,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.0,
                  "response_format": {"type": "json_object"}},
            timeout=self.timeout_s)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        import json
        raw = json.loads(content)
        probs = {q: _clamp(float(raw.get(q, 0.0))) for q in QUESTION_BANK}
        return JevDecision(
            request_id=str(uuid.uuid4()),
            timestamp=time.time(),
            probabilities=probs,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            model=self.model,
            is_mock=False,
            state_episode_tic=state.episode_tic,
            meta={"questions": list(QUESTION_BANK)},
        )


def make_jev_client(cfg: dict) -> JevClient:
    jev = cfg["jev"]
    if jev.get("mode", "mock") == "live":
        return LiveJevClient(base_url=jev.get("base_url"), model=jev.get("model"),
                             timeout_s=float(jev.get("timeout_seconds", 10.0)))
    return MockJevClient()


def _clamp(x: float) -> float:
    return float(min(1.0, max(0.0, round(x, 4))))
