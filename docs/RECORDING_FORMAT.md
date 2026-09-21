# RECORDING FORMAT v2

A recording is a directory `outputs/recordings/<run_id>/` containing:

```
<run_id>/
├── recording.jsonl     # line-delimited JSON records (this format)
└── frames/             # JPEG frames (when record_frames was enabled)
    ├── 000000.jpg
    └── ...
```

Replay requires ONLY these files — no ViZDoom, no neural simulation, no Jev
API. The FastAPI replay server (`make replay`) exposes them at:

- `GET /api/recordings` — list of runs (summary metadata each)
- `GET /api/recordings/<run_id>` — full parsed recording JSON
  (`{run_id, header, steps, frames, summary}`)
- `GET /api/recordings/<run_id>/frames/<NNNNNN>.jpg` — one JPEG frame
- `GET /api/connectome/<file>` — shared connectome viz bundle (format 2.1+,
  see "Shared connectome assets" below)

## recording.jsonl

One JSON object per line. `kind` discriminates records. The first line is
always the header; then one `step` record per controller step; then an
`episode_end` record; then `frames_meta` and `footer`.

### header (kind = "header", first line)

```json
{
  "kind": "header",
  "recording_format_version": "2.0",
  "created_at": 1790006832.4,
  "run_kind": "recorded_experiment",
  "software_version": "0.1.0",
  "seed": 42,
  "config": { "...": "full effective configuration" },
  "connectome": {
    "provenance": {
      "source": "malecns-v1.0 | fixture",
      "reduced": false,
      "note": "...",
      "retina": {"photoreceptors": 6091, "mapped": 5895}
    },
    "n_neurons": 211577,
    "n_edges": 26028386,
    "populations": {"ol_sensory": 6098, "...": 0}
  },
  "environment": {"backend": "vizdoom | fixture",
                  "scenario": "defend_the_center",
                  "actions": ["turn_left", "turn_right", "attack", "noop"]},
  "vision": {"pathway": "photoreceptor | mosaic"},
  "jev": {"mode": "mock | live", "client": "mock-jev-v1", "cadence_hz": 3.0},
  "bridge": {"mappings": {"ATTACK": "visual_projection"}, "gain": 20.0,
             "slices": {"ATTACK": {"population": "visual_projection",
                                   "start": 6900, "count": 4600}},
             "prohibited_population": "descending_neuron",
             "evidence_class": "engineering_hypothesis"},
  "motor": {"decoder": "typed_dn | population_bank", "readouts": {...},
            "contributing": {"attack": {"indices": [207651, 208123],
                                        "body_ids": [1008064715, 1008127787]}},
            "gains": {...}, "thresholds": {...}},
  "motor_population": {"indices": [173, 482, "..."],
                       "body_ids": [1000000173, "..."]},
  "connectome_assets": {"url": "/api/connectome",
                        "files": {"meta": "meta.json",
                                  "positions": "positions.f32",
                                  "population": "population.i16",
                                  "flags": "flags.u8", "ids": "ids.i64"},
                        "key": "neuron_index"},
  "telemetry": {"population_sample": 32, "top_k": 256},
  "warnings": ["MOCK JEV: ...", "REDUCED CONNECTOME: ..."]
}
```

`warnings` MUST be shown by any UI. `connectome.provenance.source` is
`"fixture"` when the synthetic test graph was used — never presented as real.

Neuron indexing contract (the frontend depends on this):

- Every neuron-level array in the shared connectome asset bundle, every index
  in `motor_population.indices`, `motor.contributing[*].indices`,
  `bridge.slices[*].start..start+count`, and every index in step-level
  `activity.top` is a NEURON INDEX — the row order of the graph arrays.
  `body_ids` (MaleCNS bodyIds) are provided alongside for cross-referencing.
- `motor_population` lists ALL descending/motor neurons (full graph: all 1314
  `descending_neuron` + all 708 `vnc_motor`, sorted by index; reduced/fixture:
  the whole motor population). Step `activity.motor_rates` is aligned to this
  exact order.
- `motor.contributing` maps each non-noop action to the exact neurons feeding
  its decoder score (typed_dn: the readout-set union per action, e.g.
  `turn_left` = DNa02 left; population_bank: that action's bank). Highlight
  these when the action fires.
- `bridge.slices` gives the contiguous neuron-index range `[start, start+count)`
  each Jev question modulates (slices never overlap; `start` is `null` when
  `count` is 0).
- `connectome_assets` is `null` for reduced/fixture recordings (no shared
  bundle; the graph is small). Otherwise see "Shared connectome assets" below.

### step (kind = "step", one per controller step)

```json
{
  "kind": "step",
  "t_ms": 1234.5,              
  "controller_step": 10,
  "episode_tic": 40,
  "frame_ref": "frames/000010.jpg",
  "state": {
    "schema_version": "1.1",
    "episode_tic": 40, "health": 96.0, "ammo": 13.0, "kills": 2,
    "enemy_visible": true, "enemy_distance": 0.55, "enemy_angle": -0.49,
    "threat_level": 0.57, "available_actions": ["..."]
  },
  "jev": {
    "request_id": "mock-000012",
    "probabilities": {"ATTACK": 0.53, "RETREAT": 0.0, "EXPLORE": 0.0,
                      "THREAT_LEVEL": 0.57, "...": "10 questions total"},
    "latency_ms": 5.1, "model": "mock-jev-v1 | jev-1.13.0", "is_mock": true,
    "state_episode_tic": 40,
    "usage": {"input_tokens": 539, "output_tokens": 89},
    "confidence": {"THREAT_LEVEL": 0.92},
    "choices": {"MOVEMENT_INTENT": "turn_to_face_enemy"}
  },
  "neural": {"time_ms": 320.0, "steps": 32, "total_spikes": 123456},
  "populations": {
    "ol_sensory": {"size": 6098, "mean_rate_hz": 1.4,
                   "sampled": [{"neuron": 12, "body_id": 10312,
                                "rate_hz": 5.2}]},
    "...": {}
  },
  "activity": {
    "top": [[173, 51.3], [1486, 51.3], [1257, 49.9]],
    "motor_rates": [0.0, 12.4, "..."]
  },
  "motor": {
    "scores": {"turn_left": 0.0, "turn_right": 0.9, "attack": 0.1, "noop": 0.0},
    "selected": "turn_right",
    "confidence": 0.83,
    "channels": {"turn": 5.4, "forward": 0.0, "attack": 1.1},
    "readout_rates": {"turn": {"positive": 45.2, "negative": 0.0},
                      "forward": {"positive": 0.0, "negative": 0.0},
                      "attack": {"positive": 1.1, "negative": 0.0}}
  },
  "reward": -0.01,
  "controller_latency_ms": 68.4
}
```

Notes for consumers:

- `jev` is `null` only before the first decision; afterwards the last valid
  decision is repeated (stale-decision semantics) — check `request_id` changes
  to detect new decisions. `probabilities` covers the full 10-question bank
  (ATTACK, RETREAT, EXPLORE, REPOSITION, SEEK_AMMO, THREAT_LEVEL,
  ENEMY_PRESENT, MOVEMENT_INTENT, TARGET_PRIORITY, ENGAGEMENT_CONFIDENCE).
  In live mode (`is_mock: false`) these come from the TypeSafe System One API
  (`POST /v1/systemone`); `model` is the server's resolved model name, `usage`
  holds input/output token counts, `confidence` the per-question confidence of
  score/choice questions, and `choices` the winning choice names. All three
  are `null` in mock mode.
- `populations` keys are coarse MaleCNS groups (`ol_sensory`,
  `visual_projection`, `visual_centrifugal`, `ol_intrinsic`, `cx_intrinsic`,
  `cb_intrinsic`, `cb_sensory`, `ascending_neuron`, `descending_neuron`,
  `vnc_sensory`, `vnc_intrinsic`, `vnc_motor`, `other`). `sampled` is a fixed
  subset (default 32/population) for heatmaps — not full activity.
- `activity` is the neuron-level data for the 3D brain view (added in 2.1):
  - `top`: the `telemetry.top_k` (default 256) most active neurons this step as
    `[neuron_index, rate_hz]` pairs, sorted by rate descending, zero-rate
    neurons excluded (so it can be shorter than top_k), rates rounded to 0.1 Hz.
  - `motor_rates`: rate in Hz (rounded to 0.1) for EVERY motor-population
    neuron, aligned element-by-element to the header's
    `motor_population.indices` (2022 values on the full graph).
  All activity is real simulated LIF activity from the recorded episode.
- `motor.channels` / `motor.readout_rates` are present for the `typed_dn`
  decoder; the `population_bank` fallback emits `raw_rates` instead.
- All rates are EMA firing rates in Hz (`rate_tau_ms` in config).

### episode_end (kind = "episode_end")

```json
{
  "kind": "episode_end",
  "ended_at": 1790006845.0,
  "episode": {
    "controller_steps": 105, "duration_s": 3.36, "total_reward": -2.19,
    "kills": 2, "health_end": 8.0, "ammo_end": 6.0,
    "action_counts": {"attack": 67, "turn_left": 38},
    "jev_decisions": 37, "jev_errors": 0,
    "controller_latency_ms": {"mean": 68.4, "p95": 75.7, "max": 112.5},
    "neural_total_spikes": 14222651
  }
}
```

### frames_meta (kind = "frames_meta")

```json
{"kind": "frames_meta", "pattern": "frames/{:06d}.jpg", "count": 105,
 "codec": "jpeg", "quality": 80}
```

In the API JSON (`GET /api/recordings/<id>`), `frames.url_pattern` is added:
`/api/recordings/<id>/frames/{:06d}.jpg`.

### footer (kind = "footer")

```json
{"kind": "footer", "ended_at": 1790006845.1}
```

## Shared connectome assets (3D brain view, format 2.1+)

Per-neuron static data (positions, ids, population labels) is exported ONCE per
machine, not per recording, to `outputs/connectome_assets/` and served at
`GET /api/connectome/<filename>` (`filename` is one of the whitelisted names
below; anything else is 400). The recording header's `connectome_assets.files`
maps logical names to filenames; all arrays are length `n_neurons` and keyed by
neuron index (the same indexing as `activity.top`, `motor_population.indices`,
`motor.contributing`, `bridge.slices`):

- `meta.json` — counts, array shapes/dtypes, population names, flag legend,
  provenance. Load this first.
- `positions.f32` — `n*3` little-endian float32 xyz soma positions (MaleCNS EM
  voxel units, 8nm). NaN = no annotated soma position (141,781 of 211,577
  neurons have one; check flag 16 or NaN before drawing).
- `population.i16` — `n` int16-LE codes into `meta.json`'s `populations` list.
- `flags.u8` — `n` bitmask: 1=photoreceptor, 2=lamina, 4=descending_neuron,
  8=vnc_motor, 16=has_position.
- `ids.i64` — `n` int64-LE MaleCNS bodyIds.

The server generates the bundle on startup from the processed full-graph cache
if it is missing; it is absent (404) on machines without the full graph —
frontends must handle that (recordings with `connectome_assets: null` never
reference it).

## Versioning

- `2.1` (current): adds header `motor_population`, `connectome_assets`,
  `telemetry`, `bridge.slices`, `motor.contributing`, and per-step `activity`
  (`top` + `motor_rates`). Additive over 2.0 — 2.0 consumers still work.
- `2.0`: RGB JPEG frames, full connectome, typed DN decoder fields,
  episode_end metrics.
- `1.0` (v1): grayscale raw `frames.u8`, reduced/fixture graph fields,
  `environment_backend` at header top level. The v1 replay page reads 1.0 only;
  the new dashboard should read 2.x (check `recording_format_version`).
