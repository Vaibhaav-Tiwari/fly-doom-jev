"""Tonic baseline calibration for the native full-connectome engine.

Motivation (chosen dynamics, NOT biology): a network sitting at rest needs a
large transient to propagate activity across synaptic hops, so visual drive
lights the optic lobe but attenuates before reaching central brain and
descending neurons. Real fly neurons have spontaneous baseline rates, and the
DOOMFLY v6 project uses tonic currents for the same reason. We calibrate a
small per-neuron tonic current per coarse population so each region's MEAN
baseline rate sits near a modest target (default 2 Hz) with no sensory input.

Uniform current would put every neuron at the same voltage and make whole
populations fire in lockstep at threshold; instead each neuron gets
``2 * mean_current * u_i`` with fixed random ``u_i ~ U[0, 1)`` (seeded), so
rates ramp smoothly with mean current and activity stays asynchronous.
A damped multiplicative controller adjusts each population's mean current
over a few settle windows; a runaway guard halves the current if a population
exceeds ``runaway_hz``.

Motor readout populations get a lower target (default 0.5 Hz) so the tonic
baseline cannot cross decoder thresholds by itself. The sensory periphery
(``ol_sensory``) is excluded by default: the retina should be quiet without
visual input.
"""

from __future__ import annotations

import logging

import numpy as np

from flydoom.malecns.full import COARSE_POPULATIONS

log = logging.getLogger(__name__)

DEFAULT_EXCLUDE = ("ol_sensory",)
DEFAULT_POP_TARGETS = {"descending_neuron": 0.5, "vnc_motor": 0.5}


def calibrate_tonic(engine, connectome, target_rate_hz: float = 2.0,
                    population_targets: dict | None = None,
                    exclude: tuple = DEFAULT_EXCLUDE,
                    max_current_mv: float = 9.0,
                    seed: int = 42, settle_ms: int = 400,
                    iterations: int = 8, runaway_hz: float = 50.0) -> dict:
    """Calibrate per-population tonic currents; returns a meta dict for the
    recording header (targets + achieved baseline rates + current stats)."""
    pop = connectome.population_index
    n = connectome.n_neurons
    targets = dict(DEFAULT_POP_TARGETS)
    targets.update(population_targets or {})
    excluded = {code for code, name in enumerate(COARSE_POPULATIONS)
                if name in exclude}
    rng = np.random.default_rng(seed)
    u = rng.random(n, dtype=np.float32)  # fixed per-neuron heterogeneity
    codes = [c for c in range(len(COARSE_POPULATIONS))
             if c not in excluded and int((pop == c).sum()) > 0]
    mean_current = {c: 0.5 for c in codes}  # start subthreshold, ramp up
    engine.reset()
    achieved: dict[str, float] = {}
    for it in range(iterations):
        tonic = np.zeros(n, dtype=np.float32)
        for c in codes:
            mask = pop == c
            tonic[mask] = 2.0 * mean_current[c] * u[mask]
        engine.set_tonic(tonic)
        engine.rate.fill(0.0)
        engine.step(settle_ms)
        rates = engine.get_population_activity(None)
        for c in codes:
            name = COARSE_POPULATIONS[c]
            r = float(rates.get(name, {}).get("mean_rate_hz", 0.0))
            target = float(targets.get(name, target_rate_hz))
            achieved[name] = round(r, 3)
            if r > runaway_hz:  # runaway guard: slash immediately
                mean_current[c] = max(0.05, mean_current[c] * 0.25)
                log.warning("tonic calibration: %s runaway (%.0f Hz), "
                            "current quartered", name, r)
            elif r <= 0.05:
                mean_current[c] = min(mean_current[c] * 2.0 + 0.25,
                                      max_current_mv)
            else:
                mean_current[c] = float(np.clip(
                    mean_current[c] * (target / max(r, 1e-6)) ** 0.5,
                    0.0, max_current_mv))
    currents = [mean_current[c] for c in codes]
    log.info("tonic baseline calibrated: target %.1f Hz, achieved %s",
             target_rate_hz, achieved)
    return {"enabled": True,
            "target_rate_hz": target_rate_hz,
            "population_targets": targets,
            "exclude": list(exclude),
            "max_current_mv": max_current_mv,
            "heterogeneity": "tonic_i = 2 * mean_current * u_i, u ~ U[0,1) "
                             f"(seed {seed}); keeps populations asynchronous",
            "mean_current_mv": {"min": round(min(currents), 3),
                                "max": round(max(currents), 3)},
            "achieved_baseline_hz": achieved,
            "evidence_class": "chosen dynamics (baseline excitability), "
                              "not measured biology"}


def maybe_calibrate_tonic(cfg: dict, connectome, engine) -> dict:
    """Hook for runner/live: calibrate when configured + native engine."""
    t = cfg.get("neural", {}).get("tonic", {})
    if not t.get("enabled", False):
        return {"enabled": False}
    from flydoom.neural.engine_native import NativeLIFEngine
    if not isinstance(engine, NativeLIFEngine):
        log.warning("tonic baseline configured but engine is not native; skipped")
        return {"enabled": False, "reason": "non-native engine"}
    return calibrate_tonic(
        engine, connectome,
        target_rate_hz=float(t.get("target_rate_hz", 2.0)),
        population_targets=t.get("population_targets"),
        exclude=tuple(t.get("exclude", DEFAULT_EXCLUDE)),
        max_current_mv=float(t.get("max_current_mv", 9.0)),
        seed=int(cfg.get("seed", 42)),
        settle_ms=int(t.get("settle_ms", 400)),
        iterations=int(t.get("iterations", 8)))
