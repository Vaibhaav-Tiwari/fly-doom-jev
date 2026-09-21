# Fly//DOOM replay

A static, replay-first React + TypeScript + Three.js experience. The fly brain is the main stage: recorded population activity lights up the 3D MaleCNS view while the matching ViZDoom frame and motor action play beside it. Click an action or population to seek and isolate its neural drive.

The browser reads static recordings directly. It never calls Jev, ViZDoom, or the simulation backend.

## Develop

```bash
cd web
npm ci
npm run dev
```

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
