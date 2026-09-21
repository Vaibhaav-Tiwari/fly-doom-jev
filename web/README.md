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

## Bundled recording limitations

The bundled samples are current v1 outputs. Their 80×60 grayscale ViZDoom buffers, state, Jev scores, sampled activity, motor scores, and timing are real recorded values. V1 does **not** include RGB, full neuron coordinates, or per-action neuron attribution, so:

- the viewport uses the recorded grayscale pixels and automatically switches to RGB when a v2 RGB shape is present;
- the brain shows the recorded sampled neurons in a clearly marked deterministic display layout until recorded coordinates exist;
- neural drive falls back only to the recording header's explicit motor source population (`descending_neuron`), never to invented attribution;
- richer region/cell-type filters appear automatically when those fields are recorded.

## Tests

```bash
npm test
npm run build
```
