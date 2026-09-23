# Fly//DOOM replay + optional live mode

A React + TypeScript + Three.js static replay experience. The fly brain is the main stage: real recorded activity lights up the 3D MaleCNS view while the matching ViZDoom frame and motor action play beside it. The featured recording opens by default and needs no backend, API key, or network service.

Live mode is optional. Set `VITE_LIVE_API` to the self-hosted broadcaster URL before building (for example `/live-api` in local development). No API key is sent to the browser. If it is not configured or reachable, the UI explains how to self-host while all recorded runs remain usable.

## Develop

```bash
cd web
npm ci
npm run dev
```

To opt into the local live broadcaster, use the included Vite proxy:

```bash
VITE_LIVE_API=/live-api npm run dev
```

The proxy forwards `/live-api` to `http://127.0.0.1:8420`. For a production build, set `VITE_LIVE_API` to the URL where the browser can reach your self-hosted broadcaster.

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

The static catalog contains three curated format 2.2 runs: a featured Jev + brain arena clear, a brain-only arena clear, and an E1M1 exit hunt with a synchronized automap. They share the full MaleCNS asset bundle: 211,577 indexed neurons, with the 141,781 annotated soma positions rendered as GPU instances. Neurons without an annotated position are never assigned invented coordinates.

## Tests

```bash
npm test
npm run build
```
