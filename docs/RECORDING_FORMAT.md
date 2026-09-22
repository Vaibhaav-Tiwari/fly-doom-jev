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
  "run_kind": "recorded_experiment",   // or "live_session" (live server episodes)
  "software_version": "0.1.0",
  "seed": 42,
  "controller": "jev",        // 'jev' (Jev+MaleCNS) | 'brain' (MaleCNS ALONE —
                              // Jev bypassed, zero API calls; 2.2+; default
                              // 'jev' when absent in older recordings)
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
                  "skill": 1,
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
  "plasticity": {"enabled": true,
                 "model": "reward-gated Hebbian on KC->MBON edges ...",
                 "plastic_edges": 61210, "kc_neurons": 4064, "mbon_neurons": 97,
                 "pam_neurons": 316, "ppl1_neurons": 16,
                 "learning_rate": 0.05, "eligibility_tau_s": 1.0,
                 "dopamine_tau_s": 1.5, "max_weight_multiplier": 3.0,
                 "clamp": "0 <= w <= max_weight_multiplier * w0 (runaway guard)",
                 "persistence": "weights persist across episodes within a run; reset on POST /new",
                 "evidence_class": "CHOSEN DYNAMICS, unvalidated ..."},
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
`environment.skill` is the ViZDoom doom_skill (1-5) the episode ran at when
the config set it explicitly (`null` = scenario default, 3 for
defend_the_center); lower is easier and must be visible alongside scores.

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
    "schema_version": "1.2",
    "episode_tic": 40, "health": 96.0, "ammo": 13.0, "kills": 2,
    "enemy_visible": true, "enemy_distance": 0.55, "enemy_angle": -0.49,
    "threat_level": 0.57, "available_actions": ["..."],
    "enemy_in_view": true, "aim_offset_deg": -44.1, "alive_s": 1.14,
    "recent_damage_taken": 4.0, "position_x": 1050.5, "position_y": -3600.0
  },
  "jev": {
    "request_id": "mock-000012",
    "probabilities": {"ATTACK": 0.53, "RETREAT": 0.0, "EXPLORE": 0.0,
                      "THREAT_LEVEL": 0.57, "...": "10 questions total"},
    "latency_ms": 5.1, "model": "mock-jev-v1 | jev-1.13.0", "is_mock": true,
    "state_episode_tic": 40,
    "usage": {"input_tokens": 539, "output_tokens": 89},
    "confidence": {"THREAT_LEVEL": 0.92},
    "choices": {"MOVEMENT_INTENT": "turn_right"},
    "choice_probabilities": {"MOVEMENT_INTENT": {"forward": 0.04, "hold": 0.14,
                                                 "turn_left": 0.33,
                                                 "turn_right": 0.45}}
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
  "retina": {"indices": [12, 340, "..."], "drive": [4.2, 3.8, "..."],
             "note": "photoreceptor input drive (mV), not spikes"},
  "motor": {
    "scores": {"turn_left": 0.0, "turn_right": 0.9, "attack": 0.1, "noop": 0.0},
    "selected": "turn_right",
    "combo": ["turn_right"],
    "confidence": 0.83,
    "channels": {"turn": 5.4, "forward": 0.0, "attack": 1.1},
    "imbalances": {"turn": 0.42, "forward": 0.0, "attack": 1.0},
    "readout_rates": {"turn": {"positive": 45.2, "negative": 0.0},
                      "forward": {"positive": 0.0, "negative": 0.0},
                      "attack": {"positive": 1.1, "negative": 0.0}}
  },
  "reward": -0.01,
  "learning": {"plastic_edges": 61210, "changed_edges": 13482,
               "mean_efficacy_rel": 0.971, "max_efficacy_rel": 1.0,
               "dopamine": -1.11, "reward_events": 126,
               "total_shaped_reward": -3.0},
  "controller_latency_ms": 68.4
}
```

Notes for consumers:

- `retina` (2.2+, full graph only; `null` for the mosaic pathway): the
  photoreceptor INPUT DRIVE for the top_k most-driven retinal neurons this
  step — `{"indices": [neuron_index...], "drive": [mV...], "note": "..."}`.
  Photoreceptors are graded-potential proxies in this model (they do not
  spike); this is luminance-derived input drive, NOT firing rate, and is
  labeled as such in the `note` field.
- `learning` (2.2+; `null` when plasticity is disabled): per-step plasticity
  stats so the UI can show the brain changing. `changed_edges` = KC->MBON
  edges whose efficacy differs from initial (|Δ| > 1e-6),
  `mean_efficacy_rel` = mean(w)/mean(w0) over plastic edges (1.0 = unchanged),
  `dopamine` = current signed dopamine trace, `reward_events` /
  `total_shaped_reward` accumulate the shaped reward (kill / health delta /
  death — the raw ViZDoom reward is living-reward only). Weights are clamped
  to [0, max_weight_multiplier*w0]; CHOSEN DYNAMICS, unvalidated (doomfly v6
  reference), NOT measured biology.
- `motor.neural_scores` / `motor.jev_weights` (2.2+; both `null` when
  `jev.action_weighting` is off): the decoder's raw scores BEFORE Jev
  weighting, and the per-action multiplier Jev applied (attack<-ATTACK —
  EXCEPT on `aim_ok` steps, where the reflex-speed rule (owner-approved
  2026-09-22) sets the attack weight to 1.0 x the INTENT posture bias so a
  clear shot never waits on the ~1.5 s strategy cadence; measured post-change
  lag ~2 controller steps (~0.2 s) — forward<-max(EXPLORE,
  MI.forward), backward<-RETREAT, turns<-max(REPOSITION,
  MI.turn_*); 1.0 when no decision). `motor.scores` is the final weighted
  score the selection came from. Chosen architecture, not biology.
- `state.schema_version` 1.2 adds the Jev-facing strategy view (additive):
  `enemy_in_view` (alias of `enemy_visible`), `aim_offset_deg`
  (`enemy_angle` x 90), `alive_s` (`episode_tic`/35),
  `recent_damage_taken` (hp lost over the last ~1 s of game time, 0 when the
  backend does not track it), and `position_x` / `position_y` (map units —
  movement/parking evidence for the unstuck reflex).
- `motor.aim_ok` (2.2+): the WORKING aim gate — true when an enemy was
  visible, inside the `attack_aim_cone_deg` cone of screen center, and within
  `attack_range` (true ViZDoom label geometry). Attack scores are zeroed when
  false; every recorded attack step should have `aim_ok: true`.
- `motor.unstuck` (2.2+): true on steps where the wall-stuck reflex forced
  the action (scenario_profiles unstuck; E1M1 default on).
- `header.scenario_profile` (2.2+): the active per-scenario control profile —
  `{"scenario", "motor": {forward_min, attack_aim_cone_deg, attack_range},
  "unstuck": {...} | null}`. Episode metrics include `unstuck_triggers` and
  `unstuck_escapes` (trigger confirmed by actual movement).
- `learning.reward_terms` + `learning.checkpoint_id` (episode-end metrics,
  2.2+): per-term event counts of the behavior shaping (`reward_attack_aimed`,
  `penalty_attack_blind`, `penalty_stuck`, `reward_escape` — chosen shaping,
  see SCIENCE.md) and the lineage id of the loaded/saved weight checkpoint
  (`null` when the fly started fresh). `header.plasticity.checkpoint_id` /
  `.checkpoint` record the same lineage; the live server persists weights to
  `outputs/learning/checkpoint.npz` (provenance-checked on load) so the fly
  keeps learning across POST /new and server restarts.
- `jev` is `null` only before the first decision; afterwards the last valid
  decision is repeated (stale-decision semantics) — check `request_id` changes
  to detect new decisions. `probabilities` covers the full 11-question bank
  (ATTACK, RETREAT, EXPLORE, REPOSITION, SEEK_AMMO, THREAT_LEVEL,
  ENEMY_PRESENT, MOVEMENT_INTENT, TARGET_PRIORITY, ENGAGEMENT_CONFIDENCE,
  INTENT). With `jev.strategy_period_s` set (live demo config), Jev runs at
  TWO cadences: fast ticks ask only `jev.fast_questions` (reflex-level,
  default ATTACK) and the full bank — including the strategic `INTENT`
  choice (engage / retreat / circle / advance / attack_now) — every
  ~1.5 s or on salient events; unasked questions carry over from the previous
  decision, so `probabilities` always covers the full bank. `meta.questions`
  (not currently recorded per step) is the honest record of what was asked.
  `choices.INTENT` is the current strategy posture; it multiplies the
  action-class weights (`motor.jev_weights` can exceed 1.0 for biased
  classes). The strategy call's state payload also carries `memory`
  (last 3 episode outcomes + current intent) — visible only in what Jev
  receives, not in the recording.
  In live mode (`is_mock: false`) these come from the TypeSafe System One API
  (`POST {JEV_BASE_URL}/systemone`; path configurable via `jev.decide_path`,
  e.g. `/decide` for the hosted proxy). `model` is the server's resolved model
  name, `usage` the token counts (plus `cost_usd` / `credits_remaining_usd`
  when the endpoint returns them — the official endpoint currently returns
  token counts only), `confidence` the per-question confidence of score/choice
  questions, `choices` the winning choice names, and `choice_probabilities`
  the full distribution over each choice question's options (great for the
  dashboard). All four are `null` in mock mode.
- `populations` keys are coarse MaleCNS groups (`ol_sensory`,
  `visual_projection`, `visual_centrifugal`, `ol_intrinsic`, `cx_intrinsic`,
  `cb_intrinsic`, `cb_sensory`, `ascending_neuron`, `descending_neuron`,
  `vnc_sensory`, `vnc_intrinsic`, `vnc_motor`, `other`). `sampled` is a fixed
  subset (default 32/population) for heatmaps — not full activity.
- `activity` is the neuron-level data for the 3D brain view (added in 2.1):
  - `top`: the `telemetry.top_k` (default 10000) most active neurons this step as
    `[neuron_index, rate_hz]` pairs, sorted by rate descending, zero-rate
    neurons excluded (so it can be shorter than top_k), rates rounded to 0.1 Hz.
  - `motor_rates`: rate in Hz (rounded to 0.1) for EVERY motor-population
    neuron, aligned element-by-element to the header's
    `motor_population.indices` (2022 values on the full graph).
  All activity is real simulated LIF activity from the recorded episode.
- `motor.channels` / `motor.readout_rates` are present for the `typed_dn`
  decoder; the `population_bank` fallback emits `raw_rates` instead.
- `motor.imbalances` (typed_dn, 2.1+): signed readout asymmetry
  (positive−negative)/(positive+negative) per channel in [−1, 1]; action
  scores are derived from imbalances (bounded) rather than raw channel
  magnitudes, so saturated rates cannot drown out the attack score.
  `null` for the population_bank decoder.
- `motor.combo` (2.1+): the list of actions whose buttons were pressed
  simultaneously this step (ViZDoom buttons are concurrent). `selected` is the
  primary action; `combo` may add one directional action co-firing with
  `attack` (typed_dn combo decoding). The population_bank decoder always emits
  a single-element combo.
- All rates are EMA firing rates in Hz (`rate_tau_ms` in config).

### episode_end (kind = "episode_end")

```json
{
  "kind": "episode_end",
  "ended_at": 1790006845.0,
  "episode": {
    "controller_steps": 105, "episode_tics": 368, "survival_s": 10.5,
    "duration_s": 3.36, "total_reward": -2.19,
    "kills": 2, "health_end": 8.0, "ammo_end": 6.0,
    "action_counts": {"attack": 67, "turn_left": 38},
    "jev_decisions": 37, "jev_errors": 0,
    "jev_disabled_reason": null,
    "jev_cost_usd_total": 0.0,
    "jev_credits_remaining_usd": null,
    "controller_latency_ms": {"mean": 68.4, "p95": 75.7, "max": 112.5},
    "neural_total_spikes": 14222651,
    "behavior": {"frac_steps_turn_saturated": 0.14,
                 "turn_reduces_aim_error_frac": 0.71,
                 "attack_when_close_frac": 0.57,
                 "attack_channel_mean_hz": 23.7},
    "learning": {"plastic_edges": 61210, "changed_edges": 13482,
                 "mean_efficacy_rel": 0.971, "max_efficacy_rel": 1.0,
                 "dopamine": 0.0, "reward_events": 126,
                 "total_shaped_reward": -3.0,
                 "delta_sha256_16": "6552bbe7a7dd29da"}
  }
}
```

`episode_tics` / `survival_s` are game time (35 tics/s); `duration_s` is
simulated neural time. `reset_reason` (live sessions only) is
`episode_finished` or `manual_reset` (POST /new aborted the episode early).
`behavior` (2.1+) summarizes control quality honestly:
turn-readout saturation fraction, fraction of turn steps that shrank the aim
error |enemy_angle| (convention-free), and how often the agent attacked when
an enemy was visible and close (< 0.4).

`learning` (2.2+; `null` when disabled) is the END-OF-EPISODE plasticity
summary: the same fields as the per-step `learning` record plus
`delta_sha256_16` — the first 16 hex chars of the sha256 over the float32
weight-delta vector of all plastic (KC->MBON) edges. Two episodes with the
same sha learned the identical weight changes; compare `changed_edges` /
`mean_efficacy_rel` across episodes to see cumulative learning (weights
persist across episodes within a live-server run unless POST /new reset them).

## featured.json (multi-episode runs)

When the runner records multiple episodes (`recording.episodes > 1` or
`--episodes N`), all episode recording directories are kept and
`outputs/recordings/featured.json` designates the best one:

```json
{"format": 1, "selection": "max survival episode_tics, then kills",
 "featured_run_id": "20260922-101530-ab12cd",
 "episodes": [{"run_id": "...", "episode_tics": 368, "kills": 2, "...": "..."}]}
```

Selection is by survival time (game tics), then kills; every episode's full
metrics are listed — nothing is hidden.

`jev_disabled_reason` is non-null ONLY when live Jev hit a permanent failure
(401/402/403) mid-run: live Jev is then disabled for the rest of the run and
this field carries the reason — the UI MUST surface it (there is never a
silent mid-recording fallback to mock). `jev_cost_usd_total` /
`jev_credits_remaining_usd` are accumulated from per-decision `usage` when the
endpoint reports costs; otherwise 0.0 / null.

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

- `2.2` (current): adds header `plasticity`, per-step `retina`,
  `motor.neural_scores` / `motor.jev_weights`, and per-step + episode-end
  `learning` (reward-modulated KC->MBON plasticity; chosen dynamics,
  unvalidated). Additive over 2.1 — 2.1 consumers still work.
- `2.1`: adds header `motor_population`, `connectome_assets`,
  `telemetry`, `bridge.slices`, `motor.contributing`, and per-step `activity`
  (`top` + `motor_rates`). Additive over 2.0 — 2.0 consumers still work.
- `2.0`: RGB JPEG frames, full connectome, typed DN decoder fields,
  episode_end metrics.
- `1.0` (v1): grayscale raw `frames.u8`, reduced/fixture graph fields,
  `environment_backend` at header top level. The v1 replay page reads 1.0 only;
  the new dashboard should read 2.x (check `recording_format_version`).
