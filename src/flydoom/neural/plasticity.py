"""Reward-modulated dopamine plasticity on mushroom-body KC->MBON synapses.

CHOSEN DYNAMICS, unvalidated — NOT measured biology (same honesty class as
the tonic baseline; the DOOMFLY v6 project is the reference implementation,
see PROVENANCE.md). The model:

1. REWARD EVENT -> DOPAMINE. A shaped game reward (kill, health delta, death —
   the raw ViZDoom reward is living-reward only, so we shape it from
   observation deltas; documented in the config) does two things:
   - injects a current pulse into the connectome's dopaminergic neurons for a
     few controller steps (PAM types for positive reward, PPL1 for negative),
     so the reward event is visible in the simulated brain itself;
   - raises a signed dopamine trace that gates the plasticity update.
2. PLASTICITY. Classic three-factor rule on KC->MBON edges only:
   an eligibility trace accumulates pre/post rate coincidences and decays;
   weight delta = learning_rate * dopamine * eligibility. Positive reward
   strengthens recently-active KC->MBON synapses, negative reward depresses
   them. Weights are clamped to [0, max_weight_multiplier * w0] — the runaway
   guard. The C LIF kernel reads the same weight array by pointer, so updates
   take effect in the live sim immediately.
3. PERSISTENCE. Weights persist across engine.reset() (episode boundaries)
   within a run — that is the point. reset() restores w0; the live server
   calls it on POST /new (fresh game requested) but NOT on natural death.

Cost: one vectorized numpy update per controller step over the KC->MBON edge
subset (tens of thousands of edges) — negligible next to the LIF step.
"""

from __future__ import annotations

import hashlib
import logging
import math

import numpy as np

log = logging.getLogger(__name__)

KC_PREFIX = "KC"
MBON_PREFIX = "MBON"
PAM_PREFIX = "PAM"      # reward-signaling dopaminergic types
PPL1_PREFIX = "PPL1"    # punishment-signaling dopaminergic types


class DopaminePlasticity:
    def __init__(self, connectome, cfg: dict):
        types = np.asarray(connectome.cell_type).astype(str)
        starts = np.char.startswith
        kc = np.flatnonzero(starts(types, KC_PREFIX))
        mbon = np.flatnonzero(starts(types, MBON_PREFIX))
        self.pam = np.flatnonzero(starts(types, PAM_PREFIX))
        self.ppl1 = np.flatnonzero(starts(types, PPL1_PREFIX))

        ptr, post = connectome.ptr, connectome.post
        is_mbon = np.zeros(connectome.n_neurons, dtype=bool)
        is_mbon[mbon] = True
        edge_chunks = []
        kc_used = []
        for i in kc:
            e0, e1 = int(ptr[i]), int(ptr[i + 1])
            if e0 == e1:
                continue
            sel = is_mbon[post[e0:e1]]
            if sel.any():
                edge_chunks.append(np.flatnonzero(sel) + e0)
                kc_used.append(i)
        if not edge_chunks:
            raise ValueError("no KC->MBON edges found in this connectome")
        self.edges = np.concatenate(edge_chunks)
        # pre index per edge: each chunk belongs to one KC, in order
        counts = np.array([len(c) for c in edge_chunks], dtype=np.int64)
        self.edge_pre = np.repeat(np.asarray(kc_used, dtype=np.int64), counts)
        self.edge_post = post[self.edges].astype(np.int64)

        self.weight = connectome.weight  # writable copy (see prepare_connectome)
        if not self.weight.flags.writeable:
            raise ValueError("connectome weight array is read-only; "
                             "prepare_connectome() must run before engine build")
        self.w0 = self.weight[self.edges].copy()
        self.w = self.w0.copy()

        self.lr = float(cfg.get("learning_rate", 0.05))
        self.elig_tau = float(cfg.get("eligibility_tau_s", 1.0))
        self.dopa_tau = float(cfg.get("dopamine_tau_s", 1.5))
        self.ref_rate = float(cfg.get("ref_rate_hz", 20.0))
        # rates below baseline_hz (tonic spontaneous activity) do NOT build
        # eligibility — otherwise the tonic baseline makes every edge
        # eligible constantly and sustained chip damage erases the whole MB
        # in minutes (measured: mean_efficacy_rel 0.58 after 3 episodes)
        self.baseline_hz = float(cfg.get("baseline_hz", 3.0))
        # depression is deliberately weaker than potentiation (asymmetric
        # gain): continuous small negative reward (chip damage) must not
        # overpower rare large positive events (kills)
        self.negative_gain = float(cfg.get("negative_gain", 0.25))
        self.pulse_mv = float(cfg.get("dan_pulse_mv", 20.0))
        self.pulse_steps = int(cfg.get("dan_pulse_steps", 3))
        self.max_mult = float(cfg.get("max_weight_multiplier", 3.0))
        self.hi = self.w0 * self.max_mult  # per-edge runaway clamp

        self.elig = np.zeros(len(self.edges), dtype=np.float64)
        self.dopa = 0.0
        self.total_reward = 0.0
        self.reward_events = 0
        self._pulse_left = 0
        self._pulse_idx: np.ndarray | None = None
        self._pulse_sign = 0.0
        log.info("plasticity: %d KC->MBON edges (%d KC, %d MBON), DANs: %d PAM "
                 "+ %d PPL1", len(self.edges), len(kc), len(mbon),
                 len(self.pam), len(self.ppl1))

    # ------------------------------------------------------------- rewards
    def note_reward(self, r: float) -> None:
        """Register a shaped reward event (called once per controller step)."""
        if r == 0.0:
            return
        self.dopa = float(np.clip(self.dopa + r, -2.0, 2.0))
        self.total_reward += float(r)
        self.reward_events += 1
        # DAN pulse: positive reward -> PAM, negative -> PPL1
        self._pulse_idx = self.pam if r > 0 else self.ppl1
        self._pulse_sign = 1.0 if r > 0 else -1.0
        self._pulse_left = self.pulse_steps

    def pending_injection(self) -> tuple[np.ndarray, np.ndarray] | None:
        """DAN pulse current for the next engine step, or None when inactive.

        Consumed once per controller step by the runner/live loop and merged
        with sensory/bridge inputs (DANs receive no other injected input).
        """
        if self._pulse_left <= 0 or self._pulse_idx is None \
                or len(self._pulse_idx) == 0:
            return None
        self._pulse_left -= 1
        cur = np.full(len(self._pulse_idx),
                      self._pulse_sign * self.pulse_mv, dtype=np.float32)
        return self._pulse_idx, cur

    # ----------------------------------------------------------- plasticity
    def update(self, rate: np.ndarray, dt_s: float) -> None:
        """Eligibility trace + dopamine-gated weight update (per ctrl step)."""
        pre_n = np.clip((rate[self.edge_pre] - self.baseline_hz)
                        / self.ref_rate, 0.0, 1.0)
        post_n = np.clip((rate[self.edge_post] - self.baseline_hz)
                         / self.ref_rate, 0.0, 1.0)
        self.elig *= math.exp(-dt_s / self.elig_tau)
        self.elig += pre_n * post_n
        self.dopa *= math.exp(-dt_s / self.dopa_tau)
        d = self.dopa if self.dopa >= 0 else self.dopa * self.negative_gain
        if abs(d) > 1e-6 and self.lr > 0:
            np.clip(self.w + self.lr * d * self.elig, 0.0, self.hi,
                    out=self.w)
            self.weight[self.edges] = self.w

    # --------------------------------------------------------------- state
    def reset(self) -> None:
        """Restore initial weights and clear all traces (fresh game)."""
        self.w[:] = self.w0
        self.weight[self.edges] = self.w0
        self.elig.fill(0.0)
        self.dopa = 0.0
        self._pulse_left = 0
        log.info("plasticity reset: weights restored to w0")

    def stats(self) -> dict:
        delta = self.w - self.w0
        changed = int(np.count_nonzero(np.abs(delta) > 1e-6))
        rel = self.w / np.maximum(self.w0, 1e-12)
        return {"plastic_edges": int(len(self.edges)),
                "changed_edges": changed,
                "mean_efficacy_rel": round(float(self.w.mean()
                                                 / max(self.w0.mean(), 1e-12)), 4),
                "max_efficacy_rel": round(float(rel.max()) if len(rel) else 1.0, 3),
                "dopamine": round(self.dopa, 4),
                "reward_events": self.reward_events,
                "total_shaped_reward": round(self.total_reward, 3)}

    def delta_sha(self) -> str:
        d = (self.w - self.w0).astype(np.float32).tobytes()
        return hashlib.sha256(d).hexdigest()[:16]

    def describe(self) -> dict:
        return {"enabled": True, "model": "reward-gated Hebbian on KC->MBON "
                "edges with eligibility traces; reward events pulse PAM (+) / "
                "PPL1 (-) dopaminergic neurons and gate the weight update",
                "plastic_edges": int(len(self.edges)),
                "kc_neurons": int(len(np.unique(self.edge_pre))),
                "mbon_neurons": int(len(np.unique(self.edge_post))),
                "pam_neurons": int(len(self.pam)),
                "ppl1_neurons": int(len(self.ppl1)),
                "learning_rate": self.lr, "eligibility_tau_s": self.elig_tau,
                "dopamine_tau_s": self.dopa_tau,
                "baseline_hz": self.baseline_hz,
                "negative_gain": self.negative_gain,
                "max_weight_multiplier": self.max_mult,
                "clamp": "0 <= w <= max_weight_multiplier * w0 (runaway guard)",
                "persistence": "weights persist across episodes within a run; "
                               "reset on POST /new (fresh game)",
                "evidence_class": "CHOSEN DYNAMICS, unvalidated — not measured "
                                  "biology (doomfly v6 reference)"}


def prepare_connectome(cfg: dict, connectome):
    """Give the connectome a writable in-RAM weight copy when plasticity is on.

    The full-graph cache is memory-mapped read-only; the C kernel binds the
    array by pointer, so this must run BEFORE engine construction. Returns the
    connectome unchanged when plasticity is disabled or the graph is not the
    full MaleCNS one.
    """
    if not cfg.get("plasticity", {}).get("enabled", False):
        return connectome
    from flydoom.malecns.full import FullConnectome
    if not isinstance(connectome, FullConnectome):
        return connectome
    if connectome.weight.flags.writeable:
        return connectome
    import dataclasses
    log.info("plasticity: copying weight array to RAM (%.0f MB) for writable "
             "KC->MBON efficacies", connectome.weight.nbytes / 1e6)
    return dataclasses.replace(connectome, weight=np.array(connectome.weight))


def maybe_make_plasticity(cfg: dict, connectome, engine):
    """Hook for runner/live: build DopaminePlasticity when configured."""
    p = cfg.get("plasticity", {})
    if not p.get("enabled", False):
        return None
    from flydoom.malecns.full import FullConnectome
    from flydoom.neural.engine_native import NativeLIFEngine
    if not isinstance(connectome, FullConnectome) \
            or not isinstance(engine, NativeLIFEngine):
        log.warning("plasticity configured but connectome/engine is not the "
                    "full native one; disabled")
        return None
    try:
        return DopaminePlasticity(connectome, p)
    except ValueError as exc:
        log.warning("plasticity disabled: %s", exc)
        return None


def shaped_reward(cfg: dict, prev_obs, obs, done: bool) -> float:
    """Shaped reward from observation deltas (config-documented).

    The raw ViZDoom reward is living-reward only in these scenarios, so
    learning-relevant events are computed here: kills, health delta (damage
    taken / pickups), death. Damage dealt is not directly observable with the
    current game variables; the kill event is its proxy (documented).
    """
    p = cfg.get("plasticity", {})
    r = 0.0
    r += float(p.get("reward_kill", 1.0)) * (obs.kills - prev_obs.kills)
    r += float(p.get("reward_health_delta", 0.02)) * (obs.health - prev_obs.health)
    if done and obs.health <= 0:
        r += float(p.get("reward_death", -1.0))
    return r
