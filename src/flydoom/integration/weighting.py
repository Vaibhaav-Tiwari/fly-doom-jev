"""Jev action-class weighting (owner-requested architecture, 2026-09-22).

The neural decoder proposes action scores from MaleCNS activity; Jev's
probabilities WEIGHT / VETO those scores per action class — Jev picks the
action class given the structured game state, the brain provides the motor
readout evidence. Weighting cannot create an action the brain did not propose
(a zero neural score stays zero); it can only amplify or veto. This is a
deliberate engineering architecture (recorded in every recording header via
`jev.action_weighting` in config + per-step `motor.jev_weights`), not biology.

Class sources:
  attack           <- ATTACK
  forward          <- max(EXPLORE, MOVEMENT_INTENT.forward)
  backward         <- RETREAT
  turn_left/right  <- max(REPOSITION, MOVEMENT_INTENT.turn_*)
"""

from __future__ import annotations

from flydoom.jev.client import JevDecision


def jev_action_weights(actions: list[str], decision: JevDecision | None,
                       floor: float = 0.0) -> dict[str, float]:
    """Per-action Jev weight in [floor, 1]; 1.0 for noop / before the first
    decision (weighting is a no-op until a real decision exists)."""
    if decision is None:
        return {a: 1.0 for a in actions}
    p = decision.probabilities
    mi = (decision.meta.get("choice_probabilities") or {}).get(
        "MOVEMENT_INTENT") or {}

    def w(action: str) -> float:
        if action == "attack":
            return max(float(p.get("ATTACK", 0.0)), floor)
        if action == "forward":
            return max(float(p.get("EXPLORE", 0.0)),
                       float(mi.get("forward", 0.0)), floor)
        if action == "backward":
            return max(float(p.get("RETREAT", 0.0)), floor)
        if action in ("turn_left", "turn_right"):
            return max(float(p.get("REPOSITION", 0.0)),
                       float(mi.get(action, 0.0)), floor)
        return 1.0

    return {a: w(a) for a in actions}


def apply_action_weighting(decoded: dict, weights: dict[str, float]) -> dict:
    """Reweight decoder scores by Jev weights and re-derive the selection.

    Keeps `neural_scores` (the raw decoder output) alongside the weighted
    `scores` so consumers can see BOTH sides of the gate. Combo decoding is
    reduced to the single weighted selection (documented in
    docs/RECORDING_FORMAT.md)."""
    raw = dict(decoded["scores"])
    weighted = {a: raw.get(a, 0.0) * weights.get(a, 1.0)
                for a in raw if a != "noop"}
    z = sum(weighted.values())
    if z <= 1e-9:
        selected, confidence = "noop", 1.0
        probs = {a: 0.0 for a in weighted}
        probs["noop"] = 1.0
    else:
        probs = {a: s / z for a, s in weighted.items()}
        probs["noop"] = 0.0
        ordered = sorted(weighted.items(), key=lambda kv: kv[1], reverse=True)
        selected = ordered[0][0]
        runner = ordered[1][1] if len(ordered) > 1 else 0.0
        confidence = float(min(1.0, (ordered[0][1] - runner)
                               / (ordered[0][1] + 1e-9)))
    out = dict(decoded)
    out["neural_scores"] = raw
    out["scores"] = probs
    out["selected"] = selected
    out["confidence"] = confidence
    out["combo"] = [] if selected == "noop" else [selected]
    out["jev_weights"] = {a: round(w, 4) for a, w in weights.items()}
    return out
