#!/usr/bin/env python
"""Benchmark the full MaleCNS neural pipeline. Measured numbers only.

Usage: python scripts/benchmark.py [--config configs/demo.yaml]

Writes results to outputs/metrics/benchmark-<timestamp>.json and prints a
summary suitable for docs/BENCHMARKS.md.
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flydoom.config import load_config  # noqa: E402
from flydoom.malecns import load_connectome  # noqa: E402
from flydoom.malecns.full import FullConnectome  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--steps", type=int, default=1000)
    args = p.parse_args()
    cfg = load_config(args.config)

    results: dict = {"platform": platform.platform(),
                     "machine": platform.machine(),
                     "python": platform.python_version(),
                     "timestamp": time.time()}

    t0 = time.perf_counter()
    conn = load_connectome(cfg)
    results["graph_load_s"] = round(time.perf_counter() - t0, 3)
    results["n_neurons"] = conn.n_neurons
    results["n_edges"] = (conn.n_edges if isinstance(conn, FullConnectome)
                          else conn.weights.nnz)
    results["connectome_source"] = conn.provenance.get("source")

    from flydoom.neural.engine_native import NativeLIFEngine
    neural = cfg["neural"]
    eng = NeuralEngine = NativeLIFEngine(
        conn, timestep_ms=float(neural.get("timestep_ms", 1.0)),
        syn_gain=float(neural.get("engine", {}).get("syn_gain", 0.3)))
    eng.inject_input(conn.retina, np.full(len(conn.retina), 20.0, dtype=np.float32))
    eng.inject_input(conn.lamina, np.full(len(conn.lamina), 8.0, dtype=np.float32))

    eng.step(100)  # warmup
    runs = []
    for batch in (100, 500, int(args.steps)):
        t0 = time.perf_counter()
        eng.step(batch)
        wall = time.perf_counter() - t0
        runs.append({"batch_steps": batch, "wall_s": round(wall, 4),
                     "sim_steps_per_s": round(batch / wall, 1),
                     "ms_per_step": round(wall * 1000 / batch, 4)})
    results["kernel"] = runs
    results["total_spikes_warmup"] = eng.total_spikes

    # vision sampling cost at demo resolution
    from flydoom.vision import PhotoreceptorPathway
    pathway = PhotoreceptorPathway(conn)
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    pathway.sample(frame, 32.0)  # warmup/prepare
    t0 = time.perf_counter()
    for _ in range(50):
        pathway.sample(frame, 32.0)
    results["vision_sample_ms"] = round((time.perf_counter() - t0) / 50 * 1000, 3)

    # population summary cost
    t0 = time.perf_counter()
    for _ in range(20):
        eng.get_population_activity(32)
    results["population_summary_ms"] = round((time.perf_counter() - t0) / 20 * 1000, 3)

    # decode cost
    from flydoom.motor import make_decoder
    dec = make_decoder(conn, cfg)
    t0 = time.perf_counter()
    for _ in range(100):
        dec.decode(eng.rate)
    results["decode_ms"] = round((time.perf_counter() - t0) / 100 * 1000, 3)

    results["peak_rss_gb"] = round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9, 2)

    out = Path("outputs/metrics") / f"benchmark-{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
