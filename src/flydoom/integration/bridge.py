"""Jev -> MaleCNS modulation bridge.

Jev probabilities are converted into input currents on UPSTREAM / INTERMEDIATE
populations only (visual_projection, ol_intrinsic, cx_intrinsic). The bridge
REFUSES to inject into the motor/readout population (descending_neuron): the
final action must emerge from network dynamics, not from Jev pressing keys.

Each question modulates a dedicated contiguous slice within its target
population, so question channels are separable and never overwrite the visual
pathway's retinal drive (the controller ADDS bridge currents on top).

Every mapping is an explicit engineering hypothesis (documented in SCIENCE.md),
not a biological claim.
"""

from __future__ import annotations

import logging

import numpy as np

from flydoom.jev.client import JevDecision
from flydoom.malecns.graph import MOTOR_POPULATION

log = logging.getLogger(__name__)


def _population_names(connectome) -> list[str]:
    from flydoom.malecns.full import COARSE_POPULATIONS, FullConnectome
    from flydoom.malecns.graph import POPULATIONS
    if isinstance(connectome, FullConnectome):
        return list(COARSE_POPULATIONS)
    return list(POPULATIONS)


class MotorInjectionError(ValueError):
    """Raised when a mapping attempts to inject Jev output into the motor/readout
    population, which would bypass the brain."""


class JevBridge:
    def __init__(self, connectome, mappings: dict[str, str],
                 gain: float = 30.0):
        self.connectome = connectome
        self.gain = float(gain)
        self.mappings: dict[str, str] = {}
        # assign each question a disjoint contiguous slice of its target population
        per_pop: dict[str, list[str]] = {}
        for question, population in mappings.items():
            self._validate(question, population)
            self.mappings[question] = population
            per_pop.setdefault(population, []).append(question)
        self._slices: dict[str, np.ndarray] = {}
        for population, questions in per_pop.items():
            idx = connectome.population_indices(population)
            slices = np.array_split(idx, max(1, len(questions)))
            for question, sl in zip(questions, slices):
                self._slices[question] = sl

    def _validate(self, question: str, population: str) -> None:
        if population not in _population_names(self.connectome):
            raise ValueError(f"unknown population {population!r} for question {question!r}")
        if population == MOTOR_POPULATION:
            raise MotorInjectionError(
                f"mapping {question!r} -> {population!r} rejected: Jev must not inject "
                "into the motor/readout population (that would bypass MaleCNS dynamics)")

    def modulation_currents(self, decision: JevDecision | None
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a Jev decision into (indices, currents).

        Every mapped slice is emitted on every call (0.0 when the question's
        probability is 0 or no decision has arrived yet), so callers can treat
        the output as a complete, overwrite-safe modulation vector.
        """
        indices: list[np.ndarray] = []
        currents: list[np.ndarray] = []
        for question, sl in self._slices.items():
            if len(sl) == 0:
                continue
            p = 0.0 if decision is None else float(decision.probabilities.get(question, 0.0))
            indices.append(sl)
            currents.append(np.full(len(sl), p * self.gain, dtype=np.float32))
        if not indices:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return np.concatenate(indices), np.concatenate(currents)

    def describe(self) -> dict:
        # Slices are contiguous ranges of neuron indices (population_indices is
        # sorted and np.array_split preserves order), so start+count is exact.
        slices = {}
        for question, sl in self._slices.items():
            slices[question] = {
                "population": self.mappings[question],
                "start": int(sl[0]) if len(sl) else None,
                "count": int(len(sl)),
            }
        return {
            "mappings": dict(self.mappings),
            "gain": self.gain,
            "slices": slices,
            "prohibited_population": MOTOR_POPULATION,
            "evidence_class": "engineering_hypothesis",
        }
