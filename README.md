# When Jev Meets a Fly in DOOM

**A real fruit-fly brain — the complete MaleCNS v1.0 connectome, 211,577
neurons and ~26M synapses, simulated as a spiking network — plays DOOM live.**
Optional strategy advice comes from **Jev** (TypeSafe System One probabilistic
decision API). Dopamine-gated plasticity means the fly keeps learning between
games. All of it is watchable on a live website.

### 🎮 Demo: https://fly-doom-jev.vercel.app

> The public site is **fully static** — it replays curated recorded episodes
> (arena clears with and without Jev, an E1M1 exit hunt) and is always up.
> **Live mode is self-hosted**: run the backend yourself (below) and the same
> frontend talks to it — bring your own machine and your own Jev API key.

---

<!-- ============================================================ -->
<!-- MEDIA: hero video. Drop a file at docs/media/hero.mp4 (or a  -->
<!-- GIF) and uncomment one of the two lines below.               -->
<!-- ============================================================ -->
<!--
https://github.com/Vaibhaav-Tiwari/fly-doom-jev/assets/hero.mp4

[![Watch the fly play DOOM](docs/media/hero-thumb.png)](docs/media/hero.mp4)
-->

<!-- MEDIA: hero screenshot — full cockpit (game + brain + panels)
![The live cockpit: DOOM feed, full connectome firing, Jev panel](docs/media/cockpit.png)
-->

## The flow

```
DOOM frame (ViZDoom, 320×240 RGB)
   │
   ▼
photoreceptor currents (retinotopic retinal interface)
   │
   ▼
MaleCNS v1.0 — 211,577 neurons, real wiring, native C LIF kernel
   │                                          ▲
   ▼                                          │ dopamine pulses on
descending-neuron readouts (BCI decoder)       │ kills / aimed shots /
   │                                           │ damage / death —
   ▼                                           │ ~61k plastic synapses,
DOOM actions (turn / move / shoot)             │ weights persist across games
   │
   ▼
Jev strategy layer (optional): structured state + persistent episode memory
→ INTENT (engage / retreat / circle / advance / attack_now) every ~1–2 s
→ biases which action class the brain's readout selects
```

One spike in the attack readout = one shot (doomfly-style hair trigger) — but
only when an enemy is truly in the reticle (real ViZDoom label geometry:
visible, inside a 14° cone, in range). Fly Arena ends when all enemies are
dead; E1M1 ends at the real exit lift, with live exit bearing/distance fed to
Jev and drawn on the site's automap.

## What you see on the site

- The actual DOOM feed, live, in color
- The **full brain** — resting neurons in solid blue, firing neurons in gold,
  decoder neurons in warm red (top 10,000 most-active per step)
- Motor output, action scores, and Jev's live choices/probabilities/latency
- **E1M1 automap** — real level geometry parsed from the WAD, with the fly's
  position and the exit lift marked, updating live
- LIVE / RECORDED, FLY ARENA / E1M1, JEV+BRAIN / BRAIN ONLY toggles

<!-- ============================================================ -->
<!-- MEDIA: screenshots gallery. Suggested shots:                 -->
<!--   1. brain firing close-up (gold on blue)                    -->
<!--   2. E1M1 with the minimap overlay                           -->
<!--   3. brain-only vs Jev comparison                            -->
<!-- ============================================================ -->
<!--
![Full connectome firing during a fight](docs/media/brain-firing.png)
![E1M1 with live automap](docs/media/e1m1-minimap.png)
![Jev probabilities mid-decision](docs/media/jev-panel.png)
-->

## Does Jev help? (measured, directional)

| Controller | Survival (early → late) | Kills/game (early → late) |
|---|---|---|
| Brain only | 19.4s → **21.6s** | 0.58 → **1.04** |
| Jev + brain | 16.6s → 15.7s | 0.67 → 0.87 |

From 488 recorded live episodes. The brain alone *learns* (dopamine
plasticity); Jev as configured adds latency and bias — the strategy layer is
deliberately being re-tuned (persistent memory, goal-bearing prompts, batch
bias search). Honest status: this is engineering on real wiring, not a
validated fly emulation — see `SCIENCE.md`.

Earlier A/B (n=3): Jev 25.8s vs heuristic 15.5s mean survival. The story
changes as the strategy layer improves; batch runner included for proper
sweeps.

## Quick start (macOS, Apple Silicon)

```bash
make setup            # venv + deps (incl. ViZDoom); compiles the C LIF kernel
make download-data    # MaleCNS v1.0 flat connectome (~1.2 GB)
make verify-data      # sha256-check against the pinned manifest
make live             # live server on 127.0.0.1:8420 (needs .env for real Jev)
make test             # 89 unit/integration tests
make batch ARGS="--seeds 42 43 --scenarios fly_arena e1m1 --controllers jev brain"
```

Jev API key: copy `.env.example` → `.env`, fill in `JEV_API_KEY`, `source
.env`. The key is used only server-side — never in recordings, logs, the repo,
or the browser. Without it, `jev.mode: mock` runs a deterministic stand-in.

Website:

```bash
cd web && npm install && npm run build
npm run preview -- --port 4173   # http://127.0.0.1:4173
```

### Self-hosting live mode

The frontend needs no backend for recorded replay. To run **live** games:

```bash
make live                                   # backend on 127.0.0.1:8420
VITE_LIVE_API=http://127.0.0.1:8420 npm run build --prefix web
```

Any machine that can run the backend (4+ GB RAM for the full connectome) can
be the live server — set `VITE_LIVE_API` to its URL at build time and the
site's LIVE toggle connects to it. Runtime consumes zero LLM tokens; only Jev
API credits are spent, server-side.

## Live API (no websockets, plain polling)

- `GET /state` — latest snapshot: JPEG frame, game stats, position, goal
  (E1M1 exit), motor scores, top-10k neuron activity, population rates,
  retinal drive, Jev block, learning stats
- `POST /new` — fresh episode; body `{"scenario": "fly_arena"|"e1m1",
  "controller": "jev"|"brain", "reset_learning": bool}`
- `GET /health` — status, uptime, episodes, Jev reachability

Runtime consumes **zero LLM tokens** — the only external service is the Jev
API (credits). Every episode is structurally recorded under
`outputs/recordings/live-*/` and replays on the site unchanged.

## Dataset export (Hugging Face-ready)

```bash
python -m flydoom.export.hf   # outputs/recordings -> outputs/hf_dataset
```

Produces a directly pushable HF dataset repo: dataset card (`README.md` with
YAML configs), `data/episodes.jsonl`, compressed `data/steps-*.jsonl.gz`
(per-step game state, motor decision, Jev block, neural totals, retinal drive,
top-K activity), and all frames with an imagefolder index. Incremental —
re-run any time to append new episodes.

## Repo layout

```
src/flydoom/
  doom/         ViZDoom wrapper + fixture env
  malecns/      full connectome builder/loader + download/verify
  neural/       native C LIF kernel + plasticity + tonic baseline + settle
  vision/       retinotopic photoreceptor pathway
  state/        typed versioned state schema (incl. E1M1 goal)
  jev/          System One client, async scheduler, question bank, memory
  integration/  Jev→MaleCNS bridge + action-class weighting
  motor/        typed DN decoder (spike-triggered attack) + reflexes
                (aim gate, aim-assist, wall-unstuck)
  telemetry/    recording writer (format 2.2)
  experiments/  closed-loop runner + headless batch sweeps
  api/          live server (FastAPI)
  export/       Hugging Face dataset exporter
  scenarios/    fly_arena WAD generator + E1M1 automap extractor
web/            the website (Three.js full-brain view, live + replay)
configs/        demo.yaml — every knob documented inline
docs/           RECORDING_FORMAT.md, BENCHMARKS.md
```

## Honesty box

The wiring is biological reconstruction data (MaleCNS v1.0). The neural
dynamics (LIF), retinal interface (inferred pixel→photoreceptor mapping),
motor decoder (typed-DN joystick mappings), Jev coupling, and reinforcement
events are **engineering models**, not measured fly physiology. Full
boundaries: `SCIENCE.md`; upstreams and licenses: `PROVENANCE.md`,
`THIRD_PARTY.md`.

## Acknowledgments

MaleCNS v1.0 (FlyWire/MRC LMB et al.) · ViZDoom · FreeDoom assets ·
inspired by [nftechie/doomfly](https://github.com/nftechie/doomfly) ·
Jev by TypeSafe AI.
