"""Native LIF engine for the full MaleCNS graph.

Compiles engine/native/lif_kernel.c with the system clang at first use (cached
by source sha256), then drives it via ctypes. Same public interface as the
pure-numpy LIFEngine used for fixture tests:

    reset() / step(n) / inject_input(indices, currents) /
    get_activity() / get_population_activity(sample) / get_motor_output()
"""

from __future__ import annotations

import ctypes as C
import hashlib
import json
import logging
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

from flydoom.malecns.full import COARSE_POPULATIONS, FullConnectome

log = logging.getLogger(__name__)

KERNEL_SOURCE = Path(__file__).resolve().parents[3] / "engine" / "native" / "lif_kernel.c"


class LifStateC(C.Structure):
    _fields_ = [
        ("n", C.c_int64),
        ("ptr", C.c_void_p), ("post", C.c_void_p), ("weight", C.c_void_p),
        ("v", C.c_void_p), ("g", C.c_void_p), ("refractory", C.c_void_p),
        ("drive", C.c_void_p), ("counts", C.c_void_p),
        ("queue", C.c_void_p), ("queue_count", C.c_void_p),
        ("cursor", C.c_int64),
        ("slots", C.c_int32), ("delay_steps", C.c_int32), ("ref_steps", C.c_int32),
        ("rest", C.c_float), ("threshold", C.c_float), ("reset", C.c_float),
        ("av", C.c_float), ("ag", C.c_float), ("gsyn", C.c_float),
    ]


def build_kernel(cache_dir: Path | None = None) -> Path:
    """Compile the kernel if needed; returns the shared library path."""
    src = KERNEL_SOURCE.read_bytes()
    sha = hashlib.sha256(src).hexdigest()
    cache_dir = cache_dir or KERNEL_SOURCE.parent / "build"
    lib = cache_dir / ("liblif.dylib" if sys.platform == "darwin" else "liblif.so")
    meta = lib.with_suffix(lib.suffix + ".json")
    if lib.exists() and meta.exists():
        record = json.loads(meta.read_text())
        if record.get("source_sha256") == sha:
            return lib
    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp = lib.with_suffix(".partial" + lib.suffix)
    subprocess.run(["clang", "-O3", "-std=c11", "-shared", "-fPIC",
                    str(KERNEL_SOURCE), "-o", str(tmp)], check=True)
    tmp.replace(lib)
    meta.write_text(json.dumps({"source_sha256": sha,
                                "flags": ["-O3", "-std=c11", "-shared", "-fPIC"]}))
    log.info("compiled %s -> %s", KERNEL_SOURCE.name, lib)
    return lib


class NativeLIFEngine:
    def __init__(self, connectome: FullConnectome, timestep_ms: float = 1.0,
                 v_rest: float = -52.0, v_threshold: float = -45.0,
                 v_reset: float = -52.0, tau_v_ms: float = 20.0,
                 tau_syn_ms: float = 5.0, syn_delay_ms: float = 2.0,
                 refractory_ms: float = 2.0, syn_gain: float = 30.0,
                 rate_tau_ms: float = 100.0):
        self.connectome = connectome
        self.n = connectome.n_neurons
        self.dt = float(timestep_ms)
        self.rate_tau = float(rate_tau_ms)

        lib = C.CDLL(str(build_kernel()))
        lib.lif_advance.argtypes = [C.POINTER(LifStateC), C.c_int64]
        lib.lif_advance.restype = None
        self._lib = lib

        self.v = np.full(self.n, v_rest, dtype=np.float32)
        self.g = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.int16)
        self.drive = np.zeros(self.n, dtype=np.float32)
        self.counts = np.zeros(self.n, dtype=np.int32)
        self.slots = int(round(syn_delay_ms / self.dt)) + 1
        self.queue = np.zeros(self.slots * self.n, dtype=np.int32)
        self.queue_count = np.zeros(self.slots, dtype=np.int32)
        self.rate = np.zeros(self.n, dtype=np.float32)

        self._state = LifStateC()
        self._state.n = self.n
        for name, arr in (("ptr", connectome.ptr), ("post", connectome.post),
                          ("weight", connectome.weight), ("v", self.v),
                          ("g", self.g), ("refractory", self.refractory),
                          ("drive", self.drive), ("counts", self.counts),
                          ("queue", self.queue), ("queue_count", self.queue_count)):
            setattr(self._state, name, _ptr(arr))
        self._state.cursor = 0
        self._state.slots = self.slots
        self._state.delay_steps = int(round(syn_delay_ms / self.dt))
        self._state.ref_steps = max(1, int(round(refractory_ms / self.dt)))
        self._state.rest = v_rest
        self._state.threshold = v_threshold
        self._state.reset = v_reset
        self._state.av = math.exp(-self.dt / tau_v_ms)
        self._state.ag = math.exp(-self.dt / tau_syn_ms)
        self._state.gsyn = syn_gain

        self.time_ms = 0.0
        self.step_count = 0
        self.total_spikes = 0

    # -- lifecycle ---------------------------------------------------------
    def reset(self) -> None:
        self.v.fill(self._state.rest)
        self.g.fill(0.0)
        self.refractory.fill(0)
        self.drive.fill(0.0)
        self.queue_count.fill(0)
        self.rate.fill(0.0)
        self._state.cursor = 0
        self.time_ms = 0.0
        self.step_count = 0
        self.total_spikes = 0

    # -- input -------------------------------------------------------------
    def inject_input(self, indices: np.ndarray, currents: np.ndarray) -> None:
        indices = np.ascontiguousarray(indices, dtype=np.int64)
        self.drive[indices] = np.asarray(currents, dtype=np.float32)

    def clear_input(self, indices: np.ndarray) -> None:
        self.drive[np.asarray(indices, dtype=np.int64)] = 0.0

    # -- dynamics ----------------------------------------------------------
    def step(self, n_steps: int = 1) -> np.ndarray:
        self.counts.fill(0)
        self._lib.lif_advance(C.byref(self._state), int(n_steps))
        elapsed_ms = n_steps * self.dt
        alpha = 1.0 - math.exp(-elapsed_ms / self.rate_tau)
        # counts over elapsed_ms -> Hz, EMA-smoothed
        self.rate += alpha * (self.counts.astype(np.float32)
                              * (1000.0 / elapsed_ms) - self.rate)
        self.time_ms += elapsed_ms
        self.step_count += n_steps
        self.total_spikes += int(self.counts.sum())
        return self.counts

    # -- readouts ----------------------------------------------------------
    def get_activity(self, indices: np.ndarray | None = None) -> np.ndarray:
        if indices is None:
            return self.rate.copy()
        return self.rate[np.asarray(indices, dtype=np.int64)].copy()

    def get_population_activity(self, sample_per_population: int | None = None
                                ) -> dict[str, dict]:
        out: dict[str, dict] = {}
        pop = self.connectome.population_index
        for code, name in enumerate(COARSE_POPULATIONS):
            idx = np.flatnonzero(pop == code)
            if len(idx) == 0:
                continue
            rates = self.rate[idx]
            entry = {"size": int(len(idx)), "mean_rate_hz": float(rates.mean())}
            if sample_per_population:
                k = min(sample_per_population, len(idx))
                sel = idx[np.linspace(0, len(idx) - 1, k).astype(np.int64)]
                entry["sampled"] = [
                    {"neuron": int(j), "body_id": int(self.connectome.ids[j]),
                     "rate_hz": float(self.rate[j])} for j in sel]
            out[name] = entry
        return out

    def get_motor_output(self) -> np.ndarray:
        return self.get_activity(self.connectome.population_indices("descending_neuron"))


def _ptr(arr: np.ndarray) -> int:
    a = np.ascontiguousarray(arr)
    return a.ctypes.data
