"""Motor decoding from neural activity.

TypedDNDecoder (full MaleCNS mode): reads known descending-neuron TYPES and
maps their population rates to joystick channels:

    turn    = gain * (rate(DNa02, R) - rate(DNa02, L))   # turning DN pair
    forward = gain * (rate(DNp09) + rate(DNg100) - rate(MDN))
    attack  = gain * rate(DNpe017)                        # BCI-style readout

Biological grounding: DNa02 turning (Rayshubskiy et al. 2020), DNp09 forward
walking, MDN backward (Carreira-Rosario et al. 2018), DNg100/BDN2 forward
(Sapkal et al. 2024). The attack readout and all gains are ENGINEERING joystick
mappings, not biology (SCIENCE.md, PROVENANCE.md). All mappings/gains are
configurable.

BankDecoder (v1) remains for graphs without type annotations (fixtures).
"""

from __future__ import annotations

import numpy as np

from flydoom.malecns.full import FullConnectome
from flydoom.malecns.graph import MOTOR_POPULATION, ReducedConnectome

DEFAULT_READOUTS = {
    "turn": {"positive": [{"type": "DNa02", "side": "R"}],
             "negative": [{"type": "DNa02", "side": "L"}]},
    "forward": {"positive": [{"type": "DNp09"}, {"type": "DNg100"}],
                "negative": [{"type": "MDN"}]},
    "attack": {"positive": [{"type": "DNpe017"}], "negative": []},
}

_SIDE_CODE = {"L": 0, "R": 1, "M": 2}


class TypedDNDecoder:
    decoder_name = "typed_dn"

    def __init__(self, connectome: FullConnectome, actions: list[str],
                 readouts: dict | None = None, turn_gain: float = 6.0,
                 forward_gain: float = 0.3, attack_gain: float = 1.0,
                 turn_min: float = 0.5, forward_min: float = 0.2,
                 attack_threshold: float = 5.0):
        self.conn = connectome
        self.actions = list(actions)
        self.readouts_cfg = readouts or DEFAULT_READOUTS
        self.gains = {"turn": float(turn_gain), "forward": float(forward_gain),
                      "attack": float(attack_gain)}
        self.mins = {"turn": float(turn_min), "forward": float(forward_min),
                     "attack": float(attack_threshold)}
        # resolve readout indices once
        self._sets: dict[str, dict[str, list[int]]] = {}
        for channel, spec in self.readouts_cfg.items():
            self._sets[channel] = {"positive": [], "negative": []}
            for sign in ("positive", "negative"):
                for r in spec.get(sign, []):
                    idx = connectome.type_indices(r["type"])
                    if "side" in r:
                        idx = idx[connectome.side[idx] == _SIDE_CODE[r["side"]]]
                    self._sets[channel][sign].extend(int(i) for i in idx)
        self.missing = {c: s for c, s in self._sets.items()
                        if not s["positive"] and not s["negative"]}

    # which readout sets feed each action's decoder score
    _ACTION_SOURCES = {"turn_left": [("turn", "negative")],
                       "turn_right": [("turn", "positive")],
                       "forward": [("forward", "positive")],
                       "backward": [("forward", "negative")],
                       "attack": [("attack", "positive")]}

    def contributing(self) -> dict:
        """Per-action neuron sets behind each decoder score (for UI highlight)."""
        out = {}
        for action in self.actions:
            if action == "noop":
                continue
            idxs: list[int] = []
            for channel, sign in self._ACTION_SOURCES.get(action, []):
                idxs.extend(self._sets.get(channel, {}).get(sign, []))
            idxs = sorted(set(idxs))
            out[action] = {"indices": idxs,
                           "body_ids": [int(self.conn.ids[i]) for i in idxs]}
        return out

    def describe(self) -> dict:
        return {"decoder": self.decoder_name,
                "readouts": {c: {s: [int(self.conn.ids[i]) for i in idxs]
                                 for s, idxs in ss.items()}
                             for c, ss in self._sets.items()},
                "contributing": self.contributing(),
                "gains": self.gains, "thresholds": self.mins,
                "evidence_class": "typed DN identities biological; channel gains "
                                  "and attack readout are engineering mappings"}

    def channels(self, rates: np.ndarray) -> dict[str, float]:
        out = {}
        for channel, ss in self._sets.items():
            pos = float(rates[ss["positive"]].sum()) if ss["positive"] else 0.0
            neg = float(rates[ss["negative"]].sum()) if ss["negative"] else 0.0
            out[channel] = (pos - neg) * self.gains[channel]
        return out

    def decode(self, full_rates: np.ndarray) -> dict:
        """full_rates: per-neuron rates for ALL neurons (engine.rate)."""
        ch = self.channels(full_rates)
        scores = {a: 0.0 for a in self.actions}
        selected, confidence = "noop", 1.0
        if "attack" in scores:
            scores["attack"] = float(max(0.0, ch.get("attack", 0.0))
                                     / max(self.mins["attack"], 1e-9))
        turn = ch.get("turn", 0.0)
        if "turn_left" in scores:
            scores["turn_left"] = float(max(0.0, -turn))
        if "turn_right" in scores:
            scores["turn_right"] = float(max(0.0, turn))
        fwd = ch.get("forward", 0.0)
        if "forward" in scores:
            scores["forward"] = float(max(0.0, fwd))
        if "backward" in scores:
            scores["backward"] = float(max(0.0, -fwd))

        # winner-take-all with thresholds; attack has priority
        if scores.get("attack", 0.0) >= 1.0:
            selected = "attack"
            confidence = min(1.0, scores["attack"] / 3.0)
        else:
            candidates = sorted(((s, a) for a, s in scores.items() if a != "noop"),
                                reverse=True)
            if candidates and candidates[0][0] > self._min_for(candidates[0][1]):
                selected = candidates[0][1]
                runner = candidates[1][0] if len(candidates) > 1 else 0.0
                confidence = float(min(1.0, (candidates[0][0] - runner)
                                       / (candidates[0][0] + 1e-9)))
        z = sum(scores.values())
        probs = {a: (s / z if z > 0 else 0.0) for a, s in scores.items()}
        probs["noop"] = 1.0 if selected == "noop" else 0.0
        return {"scores": probs, "selected": selected, "confidence": confidence,
                "channels": ch,
                "readout_rates": {c: {"positive": self._mean(ss["positive"], full_rates),
                                      "negative": self._mean(ss["negative"], full_rates)}
                                  for c, ss in self._sets.items()}}

    def _min_for(self, action: str) -> float:
        if action in ("turn_left", "turn_right"):
            return self.mins["turn"]
        if action in ("forward", "backward"):
            return self.mins["forward"]
        return 0.0

    @staticmethod
    def _mean(idxs, rates) -> float:
        return float(rates[idxs].mean()) if idxs else 0.0


class BankDecoder:
    """v1 population-bank decoder (fallback for graphs without DN types)."""

    decoder_name = "population_bank"

    def __init__(self, connectome: ReducedConnectome, actions: list[str],
                 threshold: float = 0.02):
        self.actions = list(actions)
        self.threshold = float(threshold)
        self.conn = connectome
        motor_idx = connectome.population_indices(MOTOR_POPULATION)
        if len(motor_idx) == 0:
            raise ValueError("motor population is empty")
        n_banks = len(self.actions) - 1
        banks = np.array_split(motor_idx, n_banks)
        self.banks = {a: b for a, b in zip(self.actions, banks) if a != "noop"}
        self.motor_indices = motor_idx

    def describe(self) -> dict:
        return {"decoder": self.decoder_name,
                "banks": {a: [int(i) for i in b][:8] for a, b in self.banks.items()},
                "contributing": {a: {"indices": [int(i) for i in b],
                                     "body_ids": [int(self.conn.body_ids[i]) for i in b]}
                                 for a, b in self.banks.items()},
                "evidence_class": "engineering_hypothesis (arbitrary bank split)"}

    def decode(self, full_rates: np.ndarray) -> dict:
        """full_rates: per-neuron rates for ALL neurons (engine.rate)."""
        motor_rates = full_rates[self.motor_indices]
        baseline = float(motor_rates.mean()) if len(motor_rates) else 0.0
        raw = {}
        for action, bank in self.banks.items():
            raw[action] = float(motor_rates[np.searchsorted(self.motor_indices, bank)].mean()) \
                if len(bank) else 0.0
        centered = {a: raw[a] - baseline for a in raw}
        if max(centered.values(), default=0.0) <= self.threshold:
            scores = {a: 0.0 for a in self.actions}
            scores["noop"] = 1.0
            return {"scores": scores, "selected": "noop", "confidence": 1.0,
                    "channels": {}, "raw_rates": raw}
        scale = 4.0 / (max(centered.values()) + 1e-9)
        exps = {a: float(np.exp(centered[a] * scale)) for a in centered}
        z = sum(exps.values())
        scores = {a: exps[a] / z for a in centered}
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        selected, top = ordered[0]
        second = ordered[1][1] if len(ordered) > 1 else 0.0
        scores["noop"] = 0.0
        return {"scores": scores, "selected": selected,
                "confidence": float(top - second), "channels": {}, "raw_rates": raw}


def make_decoder(connectome, cfg: dict):
    motor = cfg.get("motor", {})
    actions = motor.get("actions", ["forward", "turn_left", "turn_right",
                                    "attack", "noop"])
    if isinstance(connectome, FullConnectome) and motor.get("decoder", "typed_dn") == "typed_dn":
        dec = TypedDNDecoder(connectome, actions,
                             readouts=motor.get("readouts"),
                             turn_gain=float(motor.get("turn_gain", 6.0)),
                             forward_gain=float(motor.get("forward_gain", 0.3)),
                             attack_gain=float(motor.get("attack_gain", 1.0)),
                             turn_min=float(motor.get("turn_min", 0.5)),
                             forward_min=float(motor.get("forward_min", 0.2)),
                             attack_threshold=float(motor.get("attack_threshold", 5.0)))
        if dec.missing:
            import logging
            logging.getLogger(__name__).warning(
                "typed DN readouts missing from graph: %s", list(dec.missing))
        return dec
    return BankDecoder(connectome, actions,
                       threshold=float(motor.get("threshold", 0.02)))
