# Fly//DOOM live + replay

A React + TypeScript + Three.js experience with a live broadcast and static recorded fallback. The fly brain is the main stage: real activity lights up the 3D MaleCNS view while the matching ViZDoom frame and motor action play beside it.

Live mode talks only to the local broadcaster at `http://127.0.0.1:8420`; no API key is sent to the browser. Pressing **Play Live** requests a fresh seeded game. Every live episode is recorded by the backend and can later be served as a static replay. If the broadcaster is unavailable, the UI falls back to its bundled recordings.

## Develop

```bash
cd web
npm ci
npm run dev
```

Development mode proxies `/live-api` to port `8420`, allowing the worker preview to run on any local Vite port. A static production build connects directly to `http://127.0.0.1:8420` (the backend permits the production preview origin on port `4173`). Set `VITE_LIVE_API` at build time to override that URL.

## Build and serve

```bash
cd web
npm ci
npm run build
npm run preview
```

The static output is `web/dist/`. Vite uses relative asset URLs, so the directory can be mounted by a basic static server or by FastAPI's static-files support.

## Add a recording

1. Place the recording under `public/recordings/<id>/`.
2. Add its display metadata and JSONL path to `public/recordings/index.json`.
3. Build normally.

The loader tolerates missing optional data. It reads v1 `header`, `step`, `frames_meta`, and `footer` records and is ready for v2 RGB buffers, natural-language question arrays, aligned `frame_index` values, per-neuron positions/activity, and per-action `contributing_populations`.

## Bundled recordings

The featured recording is format 2.1 and is fully static: 93 color JPEG gameplay frames, live Jev 1.13 decisions, neuron-level activity, exact per-action decoder inputs, and the shared MaleCNS visualization bundle. The bundle indexes all 211,577 neurons; 141,781 have recorded soma coordinates and are rendered as GPU instances. Neurons without an annotated soma position remain indexed for activity and attribution but are not assigned invented coordinates.

Two legacy v1 recordings remain selectable as compatibility examples. They use recorded grayscale buffers and sampled activity. Their fallback spatial layout is explicitly labeled as an annotation.

## Tests

```bash
npm test
npm run build
```
