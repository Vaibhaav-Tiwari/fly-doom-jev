# jev-doom-fly

Closed-loop research system coupling a probabilistic decision model (**Jev**,
mocked by default) to the **full MaleCNS v1.0 *Drosophila* connectome**
(211,577 neurons / 26.0M neuron-level edges) controlling an agent in
**ViZDoom**:

```
ViZDoom RGB frame ─> retinotopic photoreceptor input ─┐
                                                      ├─> full MaleCNS LIF dynamics
typed state ─> Jev (async, mock default) ─> modulation bridge (upstream pops only) ─┘
                                                      │
        descending-neuron readouts <─ motor decoder <─┘
                      │
               ViZDoom action (turn / attack / ...)
```

Research question: can a probabilistic high-level decision signal, coupled to a
fixed biological connectome, produce observable closed-loop control behavior in
ViZDoom? See `SCIENCE.md` for exactly what is data, what is engineering, and
what is NOT claimed.

## Quick start (macOS, Apple Silicon; no CUDA)

```bash
make setup          # uv venv + deps (incl. ViZDoom); compiles the C LIF kernel on first run
make download-data  # fetch MaleCNS v1.0 flat connectome (~1.2 GB)   [if not present]
make verify-data    # sha256-check raw files against the pinned manifest
make run-demo       # record a closed-loop episode -> outputs/recordings/<id>/
make replay         # replay dashboard + API at http://127.0.0.1:8420
make live           # live mode: continuous ViZDoom+MaleCNS+Jev loop, HTTP polling API
make test           # unit + integration tests
make benchmark      # measured performance numbers -> outputs/metrics/
make batch ARGS="--seeds 42 43 --scenarios fly_arena e1m1 --controllers jev brain"
                    # batch experiments -> outputs/batch/<ts>/ (episodes.jsonl +
                    # summary.csv/json; Jev API calls only in 'jev' arms;
                    # --learn enables plasticity per arm)
```

`configs/demo.yaml` uses `jev.mode: live` — the real TypeSafe System One API
(`JEV_BASE_URL=https://api.typesafe.ai/v1`, `JEV_MODEL=jev-latest`). It needs
`JEV_API_KEY` in the environment (copy `.env.example` to `.env`, fill it in,
`source .env`; the key never reaches recordings, logs, or the browser).
Without a key, set `jev.mode: mock` for the deterministic mock — everything
else is identical.

The first `make run-demo` builds the full-graph cache from the raw feather
files (~130 s, one-time; afterwards mmap-cached, ~10 ms startup). Episodes run
in realtime (~8.75 controller steps/s; measured controller latency mean ~70 ms
— see `docs/BENCHMARKS.md`).

## What is real vs. mock vs. fixture

| Component | Status |
|---|---|
| Connectome | **FULL MaleCNS v1.0** (default). Reduced subset / deterministic fixture graph only as explicit, labeled dev/test modes |
| Synapse signs | From MaleCNS neurotransmitter predictions (ACh +, GABA −, Glu −) |
| ViZDoom | Real, headless, RGB 320×240, `defend_the_center` (default), `basic`, `deadly_corridor` |
| Visual pathway | Retinotopic: 5,895 photoreceptors mapped to ommatidia columns, bilinear sRGB sampling, lamina tonic bias |
| Motor decoder | Typed descending neurons: DNa02 turn L/R, DNp09/DNg100 forward, MDN backward, DNpe017 attack readout |
| Jev | **Mock** deterministic heuristic by default; **live** mode calls the real TypeSafe System One structured-probability API (`POST /v1/systemone`, model `jev-latest`) — typed noul/score/choice questions over the 11-question bank; two cadences: fast reflex questions (ATTACK) at 3 Hz + strategic INTENT (engage/retreat/circle/advance/attack_now) every ~1.5 s or on salient events, with last-3-episode memory in the state payload, key stays server-side (`JEV_API_KEY`, see `.env.example`) |
| Behavior honesty | Episodes score ~1–6 kills then die. Not a skilled controller; metrics recorded as-is |

## Live mode server

`make live` (with `.env` sourced) runs the real loop continuously — ViZDoom +
full MaleCNS + real Jev API — with **no agent/LLM at runtime** (only Jev API
credits). Simple HTTP polling, no websockets:

- `GET /state` — latest snapshot: status, run_id, sequence, data-url JPEG frame
  (640×480, ~6 fps cap), game stats, motor scores/channels/readouts, top-256 +
  motor neuron activity, population mean rates, Jev probabilities/choices/usage
- `POST /new` — abort the current episode, start a fresh one with a new seed.
  Optional body: `{"scenario": "fly_arena"|"e1m1"}`, `{"controller":
  "jev"|"brain"}` (brain = MaleCNS alone, zero Jev calls),
  `{"reset_learning": true}` (restart the fly's learned weights fresh — by
  default they carry over across plays and server restarts via the provenance-
  checked checkpoint `outputs/learning/checkpoint.npz`: one continuously-
  learning fly)
- `GET /health` — status, uptime, episode count, Jev reachability

Every episode is recorded in the standard format under
`outputs/recordings/live-<timestamp>-<hash>/` (`run_kind: "live_session"`), so
live sessions replay in the website unchanged. If Jev is unreachable at
startup the server keeps running on the deterministic mock in a loudly-marked
degraded mode (`status: "degraded_mock_jev"`, warning in the recording header);
`/health.jev.ok` is then `false`. CORS allows the Vite dev origins
(`localhost:4173`/`127.0.0.1:4173`). Knobs: `live.fps`, `live.frame_size`,
`live.jpeg_quality`, `api.port` in `configs/demo.yaml`; episode length comes
from `environment.max_episode_tics` / `max_controller_steps`.

## Recordings & replay

Recordings (format v2, `docs/RECORDING_FORMAT.md`) contain everything needed
to replay an episode in the browser with **no ViZDoom, no simulation, no Jev
API**: header (config, provenance, warnings), per-step state/Jev/neural/motor
records, JPEG frames, episode metrics. The frontend builds against
`docs/RECORDING_FORMAT.md`.

## Repo layout

```
src/flydoom/
  doom/         ViZDoom wrapper + deterministic fixture env (same interface)
  malecns/      full connectome builder/loader, reduced subset, fixtures, download/verify
  neural/       native C-kernel LIF engine + numpy fixture engine (same interface)
  vision/       photoreceptor retinotopic pathway + mosaic fallback
  state/        typed, versioned environment state schema
  jev/          JevClient interface, deterministic mock, async scheduler
  integration/  Jev->MaleCNS bridge (motor-injection guard)
  motor/        typed DN decoder + population-bank fallback
  telemetry/    recording writer (format v2)
  experiments/  closed-loop runner
  api/          FastAPI replay server
engine/native/  lif_kernel.c (compiled at setup, cached by sha256)
configs/        demo.yaml (+ scenario/decoder config)
docs/           RECORDING_FORMAT.md, BENCHMARKS.md
scripts/        benchmark.py
tests/          unit + integration (fixtures only; full-graph tests skip without data)
data/           manifests (raw data gitignored)
outputs/        recordings, metrics (gitignored)
```

## Documentation

- `SCIENCE.md` — data/model/engineering-assumption boundaries, limitations
- `PROVENANCE.md` — what was adapted from which upstream (DOOMFLY, FlyBrain)
- `THIRD_PARTY.md` — licenses of data/software/upstreams
- `docs/RECORDING_FORMAT.md` — recording schema (frontend contract)
- `docs/BENCHMARKS.md` — measured performance numbers

## Deferred

Real Jev plumbing (retries/caching/cost), experiment-mode comparisons, action
attribution, Three.js 3D brain viewer, Docker/public deployment. The full spec
lives in `When Jev Meets a Fly in DOOM.md`.
