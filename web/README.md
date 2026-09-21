# Recorded Experiment Replay UI

A static React + TypeScript + Three.js viewer for synchronized **Recorded Experiments**. The browser reads recording files directly; it does not call Jev, ViZDoom, or a simulation backend.

## Run locally

```bash
cd web
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`).

## Build and serve static files

```bash
cd web
npm ci
npm run build
npm run preview
```

`web/dist/` is self-contained and can be mounted by any static server or copied into a FastAPI static directory. Vite uses a relative asset base, so the build also works under a sub-path.

## Add recordings

1. Place each recording folder under `public/recordings/<id>/`.
2. Add it to `public/recordings/index.json`.
3. Point `path` at its JSONL metadata file.

The loader supports v1 JSONL (`header`, `step`, `frames_meta`, `footer`) and defensively reads anticipated v2 fields including RGB frame buffers, URL frames, `questions`, neuron coordinates, action contributions, and aligned timestamps. Missing fields render as unavailable rather than being synthesized.

## Recording limitations in the bundled samples

The two bundled recordings are current v1 research outputs. They contain 80×60 grayscale ViZDoom frames, Jev probability maps, sampled population rates, motor scores, state, and timing. The UI plays those real frames as a preloaded pixel buffer. The following v2 fields are not present and remain clearly labeled or unavailable:

- RGB game frames (the viewport will automatically use RGB when recorded)
- explicit episode-state strings
- natural-language Jev question text (v1 probability keys are shown)
- recorded 3D neuron coordinates, regions, and cell types (the 3D view uses a labeled deterministic engineering layout)
- explicit per-action contributing-population attribution (top recorded population activity is shown as decoder context)
- per-event timestamp streams separate from controller-step timestamps

No values are fabricated to fill these gaps.
