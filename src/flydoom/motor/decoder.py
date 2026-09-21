"""Configurable motor decoder: descending-neuron population activity -> actions.

The descending population is partitioned into contiguous, disjoint banks, one
per action (plus a noop reference). Action score = mean firing rate of its bank,
softmax-normalized into a probability-like score with a confidence margin. The
mapping is data-driven from the population only — no hand-picked single neurons
— and the contributing neuron indices per action are exposed for inspection.
"""

from __future__ import annotations

import numpy as np

from flydoom.malecns.graph import MOTOR_POPULATION, ReducedConnectome


class MotorDecoder:
    def __init__(self, connectome: ReducedConnectome, actions: list[str],
                 threshold: float = 0.02):
        self.actions = list(actions)
        self.threshold = float(threshold)
        motor_idx = connectome.population_indices(MOTOR_POPULATION)
        if len(motor_idx) == 0:
            raise ValueError("motor population is empty")
        n_banks = len(self.actions) - 1  # noop has no bank; it wins by default
        banks = np.array_split(motor_idx, n_banks)
        self.banks: dict[str, np.ndarray] = {
            a: b for a, b in zip(self.actions, banks) if a != "noop"}
        self.motor_indices = motor_idx

    def decode(self, motor_rates: np.ndarray) -> dict:
        """motor_rates: firing rates aligned with connectome descending order.

        Bank scores are mean-centered against the whole motor population, so the
        selection reflects RELATIVE activation (robust to global saturation).
        """
        baseline = float(motor_rates.mean()) if len(motor_rates) else 0.0
        raw = {}
        for action, bank in self.banks.items():
            raw[action] = float(motor_rates[self._bank_positions(action)].mean()) \
                if len(bank) else 0.0
        centered = {a: raw[a] - baseline for a in raw}
        if max(centered.values(), default=0.0) <= self.threshold:
            scores = {a: 0.0 for a in self.actions}
            scores["noop"] = 1.0
            return {"scores": scores, "selected": "noop", "confidence": 1.0,
                    "raw_rates": raw}
        scale = 4.0 / (max(centered.values()) + 1e-9)  # normalize before softmax
        exps = {a: float(np.exp(centered[a] * scale)) for a in centered}
        z = sum(exps.values())
        scores = {a: exps[a] / z for a in centered}
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        selected, top = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        scores["noop"] = 0.0
        return {"scores": scores, "selected": selected,
                "confidence": float(top - second), "raw_rates": raw}

    def contributing_neurons(self, action: str) -> list[int]:
        """Connectome indices of the population bank contributing to an action."""
        return self.banks.get(action, np.empty(0, dtype=np.int64)).tolist()

    def _bank_positions(self, action: str) -> np.ndarray:
        bank = self.banks[action]
        return np.searchsorted(self.motor_indices, bank)
