"""Asynchronous Jev scheduler.

Runs Jev decisions on a background thread at a fixed cadence (2-5 Hz), slower
than the game/neural loop, which NEVER blocks waiting for Jev. Stale-decision
semantics (documented in README/SCIENCE): the last valid decision stays active
until a new one arrives; ``get_decision()`` returns None only before the first
decision has ever completed.
"""

from __future__ import annotations

import logging
import threading
import time

from flydoom.state.encoder import EnvironmentState

from .client import JevAuthError, JevClient, JevDecision
from .questions import QUESTION_BANK

log = logging.getLogger(__name__)


class JevScheduler:
    """Strategy-layer scheduling on top of a JevClient.

    Two cadences (when ``strategy_period_s`` is set): FAST ticks ask only
    ``fast_questions`` (reflex-level, e.g. ATTACK) at ``cadence_hz``; the full
    question bank (including the strategic INTENT choice) runs every
    ``strategy_period_s`` seconds or immediately on a salient event (an enemy
    appearing, or a large single-step health drop). Missing questions in a
    fast decision are filled from the previous decision, so downstream
    consumers always see a constant decision shape (stale-decision semantics
    per question). ``note_episode_outcome`` keeps the last 3 episode summaries,
    sent to Jev as state.memory together with the current intent.
    """

    SALIENT_HEALTH_DROP = 15.0

    def __init__(self, client: JevClient, cadence_hz: float = 3.0,
                 fast_questions: tuple = ("ATTACK",),
                 strategy_period_s: float | None = None):
        if not (0.1 <= cadence_hz <= 100.0):  # demo target is 2-5 Hz; tests run faster
            raise ValueError("cadence_hz out of sane range")
        self.client = client
        self.period_s = 1.0 / cadence_hz
        self.fast_questions = tuple(fast_questions)
        self.strategy_period_s = strategy_period_s
        self._last_strategy = 0.0
        self._force_strategy = False
        self.episode_history: list[str] = []
        self._lock = threading.Lock()
        self._latest: JevDecision | None = None
        self._state: EnvironmentState | None = None
        self._prev_seen: EnvironmentState | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.decisions_made = 0
        self.errors = 0
        # set on permanent (auth/billing) failure: live Jev is disabled for
        # the rest of the run — never a silent mid-recording mock fallback
        self.disabled_reason: str | None = None
        self.total_cost_usd = 0.0
        self.credits_remaining_usd: float | None = None

    def update_state(self, state: EnvironmentState) -> None:
        with self._lock:
            prev = self._prev_seen
            self._prev_seen = state
            self._state = state
        # salient events force an out-of-cadence strategy call: an enemy
        # appearing, or a large single-step health drop
        if prev is not None and (
                (state.enemy_visible and not prev.enemy_visible)
                or (prev.health - state.health) >= self.SALIENT_HEALTH_DROP):
            self._force_strategy = True

    def note_episode_outcome(self, summary: str) -> None:
        """Record one line of episode history (last 3 kept) for Jev's memory."""
        self.episode_history = (self.episode_history + [summary])[-3:]

    def _memory(self) -> dict:
        with self._lock:
            latest = self._latest
        intent = (latest.meta.get("choices") or {}).get("INTENT") \
            if latest else None
        return {"recent_episodes": list(self.episode_history),
                "current_intent": intent}

    def get_decision(self) -> JevDecision | None:
        with self._lock:
            return self._latest

    def prime(self, state: EnvironmentState) -> JevDecision | None:
        """Synchronously produce the first decision (used at episode start so the
        loop never starts with an empty decision slot). Errors are swallowed the
        same way as in the background loop; permanent failures disable Jev."""
        try:
            return self._record(self.client.decide(state,
                                                   memory=self._memory()))
        except JevAuthError as exc:
            self._disable(exc)
            return None
        except Exception as exc:
            self.errors += 1
            log.warning("Jev prime failed: %s", exc)
            return None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="jev-scheduler",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _record(self, decision: JevDecision) -> JevDecision:
        with self._lock:
            if self._latest is not None and \
                    len(decision.probabilities) < len(QUESTION_BANK):
                # fast (subset) decision: carry over unasked questions so the
                # decision shape is constant (stale per-question semantics)
                prev = self._latest
                for q, p in prev.probabilities.items():
                    decision.probabilities.setdefault(q, p)
                for key in ("choices", "choice_probabilities", "confidence"):
                    merged = dict(prev.meta.get(key) or {})
                    merged.update(decision.meta.get(key) or {})
                    decision.meta[key] = merged
            self._latest = decision
        self.decisions_made += 1
        usage = decision.meta.get("usage") or {}
        self.total_cost_usd += float(usage.get("cost_usd") or 0.0)
        if usage.get("credits_remaining_usd") is not None:
            self.credits_remaining_usd = float(usage["credits_remaining_usd"])
        return decision

    def _disable(self, exc: JevAuthError) -> None:
        self.errors += 1
        self.disabled_reason = str(exc)
        log.error("LIVE JEV DISABLED for the rest of this run: %s "
                  "(recording will carry jev.disabled_reason)", exc)
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                state = self._state
            if state is not None:
                questions = None  # full bank
                if self.strategy_period_s is not None:
                    now = time.time()
                    due = (now - self._last_strategy) >= self.strategy_period_s
                    if due or self._force_strategy:
                        self._last_strategy = now
                        self._force_strategy = False
                    else:
                        questions = list(self.fast_questions)
                try:
                    self._record(self.client.decide(state, questions=questions,
                                                    memory=self._memory()))
                except JevAuthError as exc:
                    self._disable(exc)
                    break
                except Exception as exc:  # never crash the loop; keep stale decision
                    self.errors += 1
                    log.warning("Jev decision failed (keeping previous decision): %s", exc)
            self._stop.wait(self.period_s)

    # Context manager for tests
    def __enter__(self) -> "JevScheduler":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
