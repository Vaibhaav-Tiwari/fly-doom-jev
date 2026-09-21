"""Vectorized leaky integrate-and-fire (LIF) dynamics over a sparse connectome.

Design notes (v1):
- One numpy float32 vector per state variable; the sparse weight matrix is a
  scipy CSR matrix, so a step is one sparse matvec plus elementwise ops.
  No per-neuron Python objects.
- Synaptic weights are the connectome's normalized synapse counts. All
  connections are treated as excitatory in v1 — MaleCNS neurotransmitter
  predictions are not wired in yet. This is a documented limitation
  (SCIENCE.md), not hidden biology.
- "Activity" reported to telemetry/decoder is an exponential moving average
  of spike rate per neuron (alpha set by `rate_tau_ms`).
"""

from __future__ import annotations

import numpy as np

from flydoom.malecns.graph import POPULATIONS, ReducedConnectome


class LIFEngine:
    """LIF network bound to a ReducedConnectome.

    Public interface (per project spec):
        reset()
        step(n_steps=1)
        inject_input(indices, currents)
        get_activity()
        get_population_activity()
        get_motor_output()
    """

    def __init__(self, connectome: ReducedConnectome, timestep_ms: float = 2.0,
                 v_rest: float = -65.0, v_threshold: float = -50.0,
                 v_reset: float = -70.0, tau_ms: float = 10.0,
                 refractory_ms: float = 2.0, syn_scale: float = 25.0,
                 rate_tau_ms: float = 50.0, noise: float = 0.0, seed: int = 0):
        if timestep_ms <= 0 or timestep_ms > tau_ms:
            raise ValueError("timestep_ms must be in (0, tau_ms]")
        self.connectome = connectome
        self.n = connectome.n_neurons
        self.W = connectome.weights  # [post, pre] CSR float32
        self.dt = float(timestep_ms)
        self.v_rest, self.v_th, self.v_reset = v_rest, v_threshold, v_reset
        self.tau = tau_ms
        self.refractory_steps = max(1, int(round(refractory_ms / self.dt)))
        self.syn_scale = float(syn_scale)
        self.noise = float(noise)
        self._rate_alpha = 1.0 - float(np.exp(-self.dt / rate_tau_ms))
        self._rng = np.random.default_rng(seed)
        self._external = np.zeros(self.n, dtype=np.float32)
        self.time_ms = 0.0
        self.step_count = 0
        self.total_spikes = 0
        self.reset()

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self.v = np.full(self.n, self.v_rest, dtype=np.float32)
        self.spikes = np.zeros(self.n, dtype=np.float32)
        self.rate = np.zeros(self.n, dtype=np.float32)
        self._refractory = np.zeros(self.n, dtype=np.int32)
        self._external.fill(0.0)
        self.time_ms = 0.0
        self.step_count = 0
        self.total_spikes = 0

    # -- input -------------------------------------------------------------
    def inject_input(self, indices: np.ndarray, currents: np.ndarray) -> None:
        """Add external input current to specific neurons for the next step(s).

        Currents persist until overwritten (set to 0 to clear); callers refresh
        them each controller step, which keeps the fast loop simple.
        """
        indices = np.asarray(indices, dtype=np.int64)
        currents = np.asarray(currents, dtype=np.float32)
        if indices.shape != currents.shape:
            raise ValueError("indices and currents must have the same shape")
        self._external[indices] = currents

    def clear_input(self, indices: np.ndarray) -> None:
        self._external[np.asarray(indices, dtype=np.int64)] = 0.0

    # -- dynamics ----------------------------------------------------------
    def step(self, n_steps: int = 1) -> np.ndarray:
        for _ in range(n_steps):
            syn = self.W @ self.spikes  # postsynaptic drive from last-step spikes
            current = self.syn_scale * syn + self._external
            if self.noise > 0:
                current = current + self._rng.standard_normal(self.n).astype(np.float32) * self.noise
            dv = (-(self.v - self.v_rest) / self.tau + current) * self.dt
            active = self._refractory <= 0
            self.v = np.where(active, self.v + dv, self.v_reset).astype(np.float32)
            fired = (self.v >= self.v_th) & active
            self.v[fired] = self.v_reset
            self._refractory[fired] = self.refractory_steps
            self._refractory[~fired] -= 1
            self.spikes = fired.astype(np.float32)
            self.rate += self._rate_alpha * (self.spikes * (1000.0 / self.dt) - self.rate)
            self.time_ms += self.dt
            self.step_count += 1
            self.total_spikes += int(fired.sum())
        return self.spikes

    # -- readouts ----------------------------------------------------------
    def get_activity(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Firing-rate estimate (Hz) per neuron, or for the given indices."""
        if indices is None:
            return self.rate.copy()
        return self.rate[np.asarray(indices, dtype=np.int64)].copy()

    def get_population_activity(self, sample_per_population: int | None = None
                                ) -> dict[str, dict]:
        """Mean firing rate per population, plus optional sampled-neuron rates."""
        out: dict[str, dict] = {}
        for i, name in enumerate(POPULATIONS):
            idx = np.flatnonzero(self.connectome.population == i)
            if len(idx) == 0:
                out[name] = {"size": 0, "mean_rate_hz": 0.0, "sampled": []}
                continue
            rates = self.rate[idx]
            entry = {"size": int(len(idx)), "mean_rate_hz": float(rates.mean())}
            if sample_per_population:
                k = min(sample_per_population, len(idx))
                sel = idx[np.linspace(0, len(idx) - 1, k).astype(np.int64)]
                entry["sampled"] = [
                    {"neuron": int(j), "body_id": int(self.connectome.body_ids[j]),
                     "rate_hz": float(self.rate[j])} for j in sel]
            out[name] = entry
        return out

    def get_motor_output(self) -> np.ndarray:
        """Firing rates of the motor/readout population (descending neurons)."""
        from flydoom.malecns.graph import MOTOR_POPULATION
        return self.get_activity(self.connectome.population_indices(MOTOR_POPULATION))
