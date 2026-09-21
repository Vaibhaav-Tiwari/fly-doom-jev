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

from .client import JevClient, JevDecision

log = logging.getLogger(__name__)


class JevScheduler:
    def __init__(self, client: JevClient, cadence_hz: float = 3.0):
        if not (0.1 <= cadence_hz <= 100.0):  # demo target is 2-5 Hz; tests run faster
            raise ValueError("cadence_hz out of sane range")
        self.client = client
        self.period_s = 1.0 / cadence_hz
        self._lock = threading.Lock()
        self._latest: JevDecision | None = None
        self._state: EnvironmentState | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.decisions_made = 0
        self.errors = 0

    def update_state(self, state: EnvironmentState) -> None:
        with self._lock:
            self._state = state

    def get_decision(self) -> JevDecision | None:
        with self._lock:
            return self._latest

    def prime(self, state: EnvironmentState) -> JevDecision | None:
        """Synchronously produce the first decision (used at episode start so the
        loop never starts with an empty decision slot). Errors are swallowed the
        same way as in the background loop."""
        try:
            decision = self.client.decide(state)
            with self._lock:
                self._latest = decision
            self.decisions_made += 1
            return decision
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

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                state = self._state
            if state is not None:
                try:
                    decision = self.client.decide(state)
                    with self._lock:
                        self._latest = decision
                    self.decisions_made += 1
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
