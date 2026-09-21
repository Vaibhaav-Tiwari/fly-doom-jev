# jev-doom-fly (v1)

A closed-loop research demo coupling a probabilistic decision model (**Jev**,
mocked by default) to the **MaleCNS v1.0** *Drosophila* connectome (reduced
subset) driving an agent in **ViZDoom**:

```
ViZDoom ─> state / vision ─> Jev (async, mock) ─> modulation bridge ─>
MaleCNS v1.0 (REDUCED subset) LIF dynamics ─> motor decoder ─> ViZDoom action
```

Research question (long-term): can a probabilistic high-level decision signal,
coupled to a fixed biological connectome, produce observable closed-loop
control in ViZDoom? **v1 is a working end-to-end demo of the loop, not a
result.** See `SCIENCE.md` for what is data, what is engineering, and what is
not claimed.

## Quick start (macOS, Apple Silicon; no CUDA, no API key needed)

```bash
make setup      # create .venv with uv, install deps (incl. ViZDoom)
make run-demo   # record a ~20s closed-loop episode -> outputs/recordings/<id>/
make replay     # serve the replay dashboard at http://127.0.0.1:8420
make test       # unit + integration tests (deterministic fixtures)
```

`make run-demo` uses: real ViZDoom (basic scenario, headless), a real reduced
MaleCNS v1.0 subset (if the raw feather files are present — they are large and
gitignored; see "Data" below), and the deterministic mock Jev. The browser
replay needs no ViZDoom, no simulation, and no Jev API.

## Data

The reduced connectome is built from the official MaleCNS v1.0 flat-connectome
feather files (`https://male-cns.janelia.org/download/`, CC-BY 4.0):

```
data/raw/malecns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather
data/raw/malecns-v1.0/connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

Checksums are pinned in `data/manifests/malecns-v1.0.manifest.json`. The full
download/verify pipeline is **deferred past v1**; for now place the files
manually. If they are absent, `neural.mode: auto` falls back to a deterministic
**fixture graph** — always labeled as such in logs, recording headers, and the
dashboard.

## Configuration

`configs/demo.yaml` — scenario, frame skip, neural timestep, Jev cadence
(2–5 Hz), bridge mappings/gain, motor decoder actions, recording. Environment
variables `JEV_API_KEY` / `JEV_BASE_URL` / `JEV_MODEL` (see `.env.example`)
select a live Jev endpoint; v1 ships only a thin live-client stub behind the
`JevClient` interface (no retries/caching yet). The key is read server-side
only and never appears in telemetry or the browser.

## Repo layout (v1)

```
src/flydoom/
  doom/         ViZDoom wrapper + deterministic fixture env (same interface)
  malecns/      reduced connectome loader (real subset / labeled fixture)
  neural/       vectorized sparse LIF engine (reset/step/inject_input/...)
  vision/       frame -> retinal mosaic -> sensory population drive
  state/        typed, versioned environment state schema
  jev/          JevClient interface, deterministic mock, async scheduler
  integration/  Jev->MaleCNS bridge (motor-injection guard)
  motor/        population-activity motor decoder
  telemetry/    JSONL recording writer
  experiments/  closed-loop runner (make run-demo)
  api/          FastAPI replay server
web/static/     replay dashboard (plain HTML/JS, no build step)
configs/        demo.yaml
tests/          unit + integration
data/           manifests (raw data is gitignored)
outputs/        recordings (gitignored)
```

## What is real vs. fixture in v1

| Component | Status |
|---|---|
| ViZDoom environment | **Real** (headless `basic` scenario); fixture env used only in tests or explicit fallback |
| Connectome | **Real MaleCNS v1.0 reduced subset** (4 populations, ~4.4k neurons, ~122k edges) when raw data present; labeled fixture graph otherwise |
| Neuron model | LIF on normalized synapse counts; excitatory-only (v1 limitation) |
| Jev | **Mock** (deterministic heuristic); live API client is a stub |
| Replay dashboard | Real; replays recordings without any live services |

## Deferred beyond v1

Full-scale connectome + download/verify pipeline, real Jev plumbing
(retries/caching/cost), experiment modes, action attribution, Three.js 3D brain
viewer, benchmarks, Docker, public deployment. The full spec lives in
`When Jev Meets a Fly in DOOM.md`.

## Provenance & licenses

- MaleCNS v1.0: Janelia FlyEM / Cambridge / MRC LMB / Google Research, CC-BY 4.0
- ViZDoom: Farama Foundation, MIT
- Upstream inspiration reviewed (no code vendored): DOOMFLY (MIT),
  fly-connectome-template (Cobanov Template Attribution License 1.0 — requires
  web-UI attribution if substantial portions are reused; v1 vendors none),
  FlyBrain (MIT).
