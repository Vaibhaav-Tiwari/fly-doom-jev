# BENCHMARKS.md — measured numbers only

All numbers below were measured on this repository's code by
`scripts/benchmark.py` (raw JSON in `outputs/metrics/benchmark-*.json`).
Nothing here is estimated or extrapolated.

## Machine

- macOS 26.4, Apple Silicon (arm64), Python 3.14.7, clang 21 (Apple).
- No CUDA/GPU acceleration is used; the neural kernel is single-threaded C.

## Full MaleCNS v1.0 graph

- Neurons: **211,577** (all annotated bodies)
- Neuron-level edges: **26,028,386** (segment-level edges with unannotated
  endpoints dropped — the standard neuron-level projection)
- Cache build time (one-time, from raw feather): **~130 s**, peak RSS ~8 GB
- Cache load (memory-mapped .npy): **0.008 s**; runtime RSS **~0.3 GB**
  (graph arrays stay in the OS page cache)

## Neural kernel throughput (`engine/native/lif_kernel.c`, dt = 1 ms)

Measured with full retinal + lamina drive (active network, millions of
spikes per batch — this is the loaded case, not the idle one):

| Batch | Sim steps/s | ms per sim step |
|---|---|---|
| 100 steps  | 828.5 | 1.207 |
| 500 steps  | 754.2 | 1.326 |
| 1000 steps | 743.3 | 1.345 |

- Interpretation: ~0.75–0.83× real-time at 1 ms resolution under load, i.e.
  one 32 ms neural interval (one controller step) costs ~43 ms of wall time.
  Idle/quiescent-network throughput measured earlier at ~3,900 steps/s
  (0.26 ms/step); the loaded number above is the honest one for gameplay.

## Per-controller-step costs (320×240 RGB input)

| Component | Mean time |
|---|---|
| Retinotopic photoreceptor sampling (5,895 receptors, bilinear RGB) | 1.05 ms |
| Population telemetry summary (13 groups + sampling) | 1.23 ms |
| Typed DN decode | 0.013 ms |
| **Whole controller step in a real episode** (incl. 32 neural steps + ViZDoom) | **mean 68–74 ms, p95 ~76–85 ms, max 120 ms** |

The realtime demo paces controller steps at 8.75 Hz (114 ms budget); the
measured mean fits, the p95/max occasionally exceed it slightly (recorded in
each episode's metrics).

## Episode metrics (recorded, `outputs/recordings/`)

From the shipped demo recordings (defend_the_center, full graph, mock Jev):
episodes last ~70–130 controller steps (~8–15 s) and score **1–6 kills**
before the player dies. These are honest, unrewarded-engineered numbers — the
controller is not good at DOOM, and we say so.

## Reproduce

```bash
make benchmark   # writes outputs/metrics/benchmark-<ts>.json
```
