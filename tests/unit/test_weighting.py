"""Jev action weighting: weights/veto semantics."""

from flydoom.integration.weighting import (apply_action_weighting,
                                           jev_action_weights)


def _decision(probs, mi=None):
    class D:
        probabilities = probs
        meta = {"choice_probabilities": {"MOVEMENT_INTENT": mi or {}}}

    return D()


ACTIONS = ["forward", "backward", "turn_left", "turn_right", "attack", "noop"]


def test_no_decision_is_noop_weighting():
    w = jev_action_weights(ACTIONS, None)
    assert all(v == 1.0 for v in w.values())


def test_weights_follow_probabilities():
    d = _decision({"ATTACK": 0.9, "EXPLORE": 0.3, "RETREAT": 0.1,
                   "REPOSITION": 0.4}, mi={"forward": 0.6, "turn_left": 0.5})
    w = jev_action_weights(ACTIONS, d)
    assert w["attack"] == 0.9
    assert w["forward"] == 0.6          # max(EXPLORE 0.3, MI.forward 0.6)
    assert w["backward"] == 0.1
    assert w["turn_left"] == 0.5        # max(REPOSITION 0.4, MI.turn_left 0.5)
    assert w["noop"] == 1.0


def test_weighting_vetoes_and_reselects():
    decoded = {"scores": {"attack": 0.1, "forward": 0.8, "noop": 0.1},
               "selected": "forward", "confidence": 0.7, "combo": ["forward"]}
    out = apply_action_weighting(decoded, {"attack": 1.0, "forward": 0.0,
                                           "noop": 1.0})
    assert out["selected"] == "attack"       # forward vetoed to zero
    assert out["combo"] == ["attack"]
    assert out["neural_scores"] == decoded["scores"]  # raw preserved
    assert out["jev_weights"]["forward"] == 0.0


def test_weighting_all_zero_is_noop():
    decoded = {"scores": {"attack": 0.5, "forward": 0.5}, "selected": "attack",
               "confidence": 0.5, "combo": ["attack"]}
    out = apply_action_weighting(decoded, {"attack": 0.0, "forward": 0.0})
    assert out["selected"] == "noop"
    assert out["combo"] == []
