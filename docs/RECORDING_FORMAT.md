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
  "bridge": {"mappings": {"ATTACK": "visual_projection"}, "gain": 12.0,
             "prohibited_population": "descending_neuron",
             "evidence_class": "engineering_hypothesis"},
  "motor": {"decoder": "typed_dn | population_bank", "readouts": {...},
            "gains": {...}, "thresholds": {...}},
  "warnings": ["MOCK JEV: ...", "REDUCED CONNECTOME: ..."]
}
```

`warnings` MUST be shown by any UI. `connectome.provenance.source` is
`"fixture"` when the synthetic test graph was used — never presented as real.

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
                      "THREAT_LEVEL": 0.57},
    "latency_ms": 5.1, "model": "mock-jev-v1", "is_mock": true,
    "state_episode_tic": 40
  },
  "neural": {"time_ms": 320.0, "steps": 32, "total_spikes": 123456},
  "populations": {
    "ol_sensory": {"size": 6098, "mean_rate_hz": 1.4,
                   "sampled": [{"neuron": 12, "body_id": 10312,
                                "rate_hz": 5.2}]},
    "...": {}
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
  to detect new decisions.
- `populations` keys are coarse MaleCNS groups (`ol_sensory`,
  `visual_projection`, `visual_centrifugal`, `ol_intrinsic`, `cx_intrinsic`,
  `cb_intrinsic`, `cb_sensory`, `ascending_neuron`, `descending_neuron`,
  `vnc_sensory`, `vnc_intrinsic`, `vnc_motor`, `other`). `sampled` is a fixed
  subset (default 32/population) for heatmaps — not full activity.
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

## Versioning

- `2.0` (current): RGB JPEG frames, full connectome, typed DN decoder fields,
  episode_end metrics.
- `1.0` (v1): grayscale raw `frames.u8`, reduced/fixture graph fields,
  `environment_backend` at header top level. The v1 replay page reads 1.0 only;
  the new dashboard should read 2.0 (check `recording_format_version`).
