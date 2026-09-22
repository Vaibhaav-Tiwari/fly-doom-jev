"""Unit tests for reward-modulated dopamine plasticity (synthetic graph only)."""

from __future__ import annotations

import numpy as np
import pytest

from flydoom.neural.plasticity import (DopaminePlasticity, prepare_connectome,
                                       shaped_reward)


class FakeConnectome:
    """Minimal stand-in with the fields DopaminePlasticity reads."""

    def __init__(self):
        # 7 neurons: 0,1 KC (with edges); 2 MBON01; 3 PAM01-a; 4 PPL101;
        # 5 other; 6 KC with NO outgoing edges (regression: must not offset
        # the edge_pre mapping)
        self.n_neurons = 7
        self.cell_type = np.array(["KCg", "KCa'b'-m", "MBON01", "PAM01",
                                   "PPL101", "L1", "KCab-p"], dtype="U10")
        # CSR by pre: 0->2 (w .5), 0->5 (w .3); 1->2 (w .4); 2->5; 3->2; 4->2; 5->; 6->
        self.ptr = np.array([0, 2, 3, 4, 5, 6, 6, 6], dtype=np.int64)
        self.post = np.array([2, 5, 2, 5, 2, 2], dtype=np.int32)
        self.weight = np.array([0.5, 0.3, 0.4, 0.1, 0.2, 0.2], dtype=np.float32)


CFG = {"learning_rate": 0.5, "eligibility_tau_s": 1.0, "dopamine_tau_s": 1.5,
       "ref_rate_hz": 20.0, "dan_pulse_mv": 20.0, "dan_pulse_steps": 2,
       "max_weight_multiplier": 3.0}


def make():
    return DopaminePlasticity(FakeConnectome(), CFG)


def test_edge_extraction_kc_to_mbon_only():
    p = make()
    # only edges 0 (KCg->MBON01) and 2 (KCa'b'-m->MBON01) are plastic
    assert sorted(p.edges.tolist()) == [0, 2]
    assert p.edge_post.tolist() == [2, 2]
    assert len(p.pam) == 1 and len(p.ppl1) == 1


def test_positive_reward_strengthens_active_edges():
    p = make()
    rate = np.full(7, 20.0, dtype=np.float32)  # all at ref rate
    p.update(rate, 0.1)                        # build eligibility, no dopamine
    assert np.allclose(p.weight[p.edges], p.w0)  # no reward -> no change
    p.note_reward(1.0)
    p.update(rate, 0.1)
    assert (p.weight[p.edges] > p.w0).all()
    # non-plastic weights untouched
    assert p.weight[1] == pytest.approx(0.3)
    assert p.weight[3] == pytest.approx(0.1)


def test_negative_reward_depresses_and_clamps_at_zero():
    p = make()
    rate = np.full(7, 20.0, dtype=np.float32)
    for _ in range(50):  # heavy negative reward must not go below 0
        p.note_reward(-1.0)
        p.update(rate, 0.1)
    assert (p.weight[p.edges] >= 0.0).all()
    assert (p.weight[p.edges] < p.w0).all()


def test_runaway_clamp_at_max_multiplier():
    p = make()
    rate = np.full(7, 100.0, dtype=np.float32)
    for _ in range(200):
        p.note_reward(1.0)
        p.update(rate, 0.1)
    assert (p.weight[p.edges] <= p.w0 * 3.0 + 1e-6).all()
    assert (p.weight[p.edges] > p.w0).all()  # and it did grow


def test_no_change_without_dopamine():
    p = make()
    rate = np.full(7, 20.0, dtype=np.float32)
    for _ in range(10):
        p.update(rate, 0.1)
    assert np.allclose(p.weight[p.edges], p.w0)


def test_dan_pulse_sign_and_duration():
    p = make()
    assert p.pending_injection() is None
    p.note_reward(1.0)
    idx, cur = p.pending_injection()
    assert idx.tolist() == p.pam.tolist() and (cur > 0).all()
    assert p.pending_injection() is not None      # 2nd of 2 pulse steps
    assert p.pending_injection() is None          # exhausted
    p.note_reward(-0.5)
    idx, cur = p.pending_injection()
    assert idx.tolist() == p.ppl1.tolist() and (cur < 0).all()


def test_reset_restores_weights():
    p = make()
    rate = np.full(7, 20.0, dtype=np.float32)
    p.note_reward(1.0)
    p.update(rate, 0.1)
    assert not np.allclose(p.weight[p.edges], p.w0)
    p.reset()
    assert np.allclose(p.weight[p.edges], p.w0)
    assert p.stats()["changed_edges"] == 0


def test_stats_and_delta_sha():
    p = make()
    s0 = p.stats()
    assert s0["changed_edges"] == 0 and s0["plastic_edges"] == 2
    rate = np.full(7, 20.0, dtype=np.float32)
    p.note_reward(1.0)
    p.update(rate, 0.1)
    s1 = p.stats()
    assert s1["changed_edges"] == 2
    assert s1["mean_efficacy_rel"] > 1.0
    assert len(p.delta_sha()) == 16
    assert p.delta_sha() != DopaminePlasticity(FakeConnectome(), CFG).delta_sha()


def test_readonly_weight_array_rejected():
    conn = FakeConnectome()
    conn.weight.setflags(write=False)
    with pytest.raises(ValueError, match="read-only"):
        DopaminePlasticity(conn, CFG)


def test_shaped_reward_from_observation_deltas():
    from flydoom.doom.base import Observation
    cfg = {"plasticity": {"reward_kill": 1.0, "reward_health_delta": 0.02,
                          "reward_death": -1.0}}

    def obs(kills, health):
        return Observation(frame=np.zeros((2, 2, 3), np.uint8), health=health,
                           ammo=10.0, kills=kills, position_x=0, position_y=0,
                           angle_deg=0, enemy_visible=False, enemy_distance=1,
                           enemy_angle=0, episode_tic=0)
    a, b = obs(0, 100.0), obs(1, 70.0)
    assert shaped_reward(cfg, a, b, done=False) == pytest.approx(1.0 - 0.6)
    assert shaped_reward(cfg, a, obs(0, 0.0), done=True) == pytest.approx(-3.0)
    assert shaped_reward(cfg, a, obs(0, 100.0), done=False) == 0.0
