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


_SIDE_CODE = {"L": 0, "R": 1, "M": 2}


class MotorInjectionError(ValueError):
    """Raised when a mapping attempts to inject Jev output into the motor/readout
    population, which would bypass the brain."""


class JevBridge:
    """Scalar question probabilities modulate population slices; choice
    questions (e.g. MOVEMENT_INTENT) additionally modulate per-choice slices
    via their full probability distribution (decision.meta.choice_probabilities)
    — e.g. turn_left/turn_right bias opposite visual_projection hemispheres.
    All targets are upstream/intermediate populations; the motor guard applies
    to both mapping kinds."""

    def __init__(self, connectome, mappings: dict[str, str],
                 gain: float = 30.0,
                 choice_mappings: dict[str, dict[str, dict]] | None = None):
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
        # choice mappings: {question: {choice: {"population"|"type": name,
        #                                        "side": "L"|"R"|null}}}
        self.choice_mappings = choice_mappings or {}
        self._choice_slices: dict[str, dict[str, np.ndarray]] = {}
        for question, choices in self.choice_mappings.items():
            self._choice_slices[question] = {}
            for choice, spec in choices.items():
                self._choice_slices[question][choice] = \
                    self._resolve_choice_target(f"{question}:{choice}", spec)

    def _validate(self, question: str, population: str) -> None:
        if population not in _population_names(self.connectome):
            raise ValueError(f"unknown population {population!r} for question {question!r}")
        if population == MOTOR_POPULATION:
            raise MotorInjectionError(
                f"mapping {question!r} -> {population!r} rejected: Jev must not inject "
                "into the motor/readout population (that would bypass MaleCNS dynamics)")

    def _resolve_choice_target(self, label: str, spec: dict) -> np.ndarray:
        """Resolve {"population": name} or {"type": cell_type} (+optional side)
        to neuron indices. Guard is by MEMBERSHIP: any target overlapping the
        motor/readout population is rejected, regardless of how it was named."""
        from flydoom.malecns.full import FullConnectome
        if "type" in spec:
            if not isinstance(self.connectome, FullConnectome):
                raise ValueError(f"{label}: type targets require the full connectome")
            idx = self.connectome.type_indices(spec["type"])
        else:
            self._validate(label, spec["population"])
            idx = self.connectome.population_indices(spec["population"])
        side = spec.get("side")
        if side is not None:
            if not isinstance(self.connectome, FullConnectome):
                raise ValueError(f"{label}: side-selective targets require the "
                                 "full connectome (side annotations)")
            idx = idx[self.connectome.side[idx] == _SIDE_CODE[side]]
        motor = set(int(i) for i in self.connectome.population_indices(MOTOR_POPULATION))
        if any(int(i) in motor for i in idx):
            raise MotorInjectionError(
                f"mapping {label!r} -> {spec} rejected: target overlaps the "
                "motor/readout population (that would bypass MaleCNS dynamics)")
        return idx

    def modulation_currents(self, decision: JevDecision | None
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Convert a Jev decision into (indices, currents).

        Every mapped slice is emitted on every call (0.0 when the question's
        probability is 0 or no decision has arrived yet), so callers can treat
        the output as a complete, overwrite-safe modulation vector. Choice
        slices are driven by the per-choice probability distribution when the
        decision carries one (meta.choice_probabilities).
        """
        indices: list[np.ndarray] = []
        currents: list[np.ndarray] = []
        for question, sl in self._slices.items():
            if len(sl) == 0:
                continue
            p = 0.0 if decision is None else float(decision.probabilities.get(question, 0.0))
            indices.append(sl)
            currents.append(np.full(len(sl), p * self.gain, dtype=np.float32))
        choice_dists = {} if decision is None else (decision.meta.get("choice_probabilities") or {})
        for question, choices in self._choice_slices.items():
            dist = choice_dists.get(question) or {}
            for choice, sl in choices.items():
                if len(sl) == 0:
                    continue
                p = float(dist.get(choice, 0.0))
                indices.append(sl)
                currents.append(np.full(len(sl), p * self.gain, dtype=np.float32))
        if not indices:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return np.concatenate(indices), np.concatenate(currents)

    def describe(self) -> dict:
        # Slices are contiguous ranges of neuron indices (population_indices is
        # sorted and np.array_split preserves order), so start+count is exact.
        def _range(sl) -> dict:
            return {"start": int(sl[0]) if len(sl) else None,
                    "count": int(len(sl))}

        slices = {}
        for question, sl in self._slices.items():
            slices[question] = {"population": self.mappings[question],
                                **_range(sl)}
        choice_slices = {}
        for question, choices in self._choice_slices.items():
            for choice, sl in choices.items():
                spec = self.choice_mappings[question][choice]
                target = {"type": spec["type"]} if "type" in spec \
                    else {"population": spec["population"]}
                choice_slices[f"{question}:{choice}"] = {
                    **target, "side": spec.get("side"), **_range(sl)}
        return {
            "mappings": dict(self.mappings),
            "gain": self.gain,
            "slices": slices,
            "choice_slices": choice_slices,
            "prohibited_population": MOTOR_POPULATION,
            "evidence_class": "engineering_hypothesis",
        }
