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


def test_intent_bias_amplifies_posture():
    d = _decision({"ATTACK": 0.5, "EXPLORE": 0.4, "RETREAT": 0.4,
                   "REPOSITION": 0.2})
    d.meta["choices"] = {"INTENT": "retreat"}
    w = jev_action_weights(ACTIONS, d)
    assert w["backward"] == 0.4 * 1.6          # RETREAT x retreat bias
    assert w["forward"] == 0.4 * 0.5           # forward suppressed in retreat
    assert w["turn_left"] == 0.2 * 1.2
    assert w["attack"] == 0.5 * 0.3            # retreat suppresses attack too


def test_intent_bias_config_override():
    d = _decision({"ATTACK": 0.5})
    d.meta["choices"] = {"INTENT": "attack_now"}
    w = jev_action_weights(ACTIONS, d,
                           intent_biases={"attack_now": {"attack": 3.0}})
    assert w["attack"] == 1.5


def test_reflex_speed_attack_when_aim_ok():
    d = _decision({"ATTACK": 0.05})  # stale/low ATTACK probability
    w = jev_action_weights(ACTIONS, d, aim_ok=True)
    assert w["attack"] == 1.0   # clear shot must not wait on Jev
    w2 = jev_action_weights(ACTIONS, d, aim_ok=False)
    assert w2["attack"] == 0.05  # unchanged when not aimed


def test_reflex_attack_intent_still_modulates():
    d = _decision({"ATTACK": 0.05})
    d.meta["choices"] = {"INTENT": "attack_now"}
    assert jev_action_weights(ACTIONS, d, aim_ok=True)["attack"] == 1.8
    d.meta["choices"] = {"INTENT": "retreat"}
    assert jev_action_weights(ACTIONS, d, aim_ok=True)["attack"] == 0.3
