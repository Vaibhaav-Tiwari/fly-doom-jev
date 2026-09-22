"""HF dataset export: recordings -> pushable dataset directory."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from flydoom.export.hf import export


def _fake_recording(root: Path, name: str, controller: str) -> Path:
    d = root / name
    (d / "frames").mkdir(parents=True)
    (d / "frames" / "000000.jpg").write_bytes(b"\xff\xd8\xff")  # JPEG magic
    lines = [
        {"kind": "header", "recording_format_version": "2.2",
         "created_at": 1.0, "run_kind": "live_session", "seed": 7,
         "controller": controller,
         "config": {"environment": {"scenario": "fly_arena", "skill": 1}}},
        {"kind": "step", "t_ms": 100.0, "controller_step": 0, "episode_tic": 4,
         "frame_ref": "frames/000000.jpg",
         "state": {"health": 100.0, "ammo": 50.0, "kills": 0,
                   "enemy_visible": True, "enemy_distance": 0.3,
                   "enemy_angle": 0.05, "aim_offset_deg": 4.5,
                   "threat_level": 0.5, "position_x": 1.0, "position_y": 2.0,
                   "alive_s": 0.0},
         "motor": {"scores": {"attack": 1.0}, "selected": "attack",
                   "combo": ["attack"], "confidence": 0.5, "aim_ok": True,
                   "attack_spiked": True, "attack_assisted": False,
                   "unstuck": False, "jev_weights": None},
         "jev": {"intent": "engage", "probabilities": {"ATTACK": 0.8}}
                if controller == "jev" else None,
         "neural": {"time_ms": 96.0, "steps": 96, "total_spikes": 12345},
         "retina": {"indices": [1, 2], "drive": [0.5, 0.6]},
         "activity": {"top": [[1, 99.0], [2, 50.0]],
                      "populations": {"optic": 12.0}}},
        {"kind": "episode_end", "ended_at": 2.0,
         "episode": {"controller_steps": 1, "controller": controller,
                     "survival_s": 1.5, "duration_s": 1.4, "kills": 1,
                     "health_end": 0.0, "ammo_end": 49.0,
                     "total_reward": 1.0, "reset_reason": "died",
                     "jev_decisions": 1, "jev_tokens": 100}},
        {"kind": "footer"},
    ]
    with (d / "recording.jsonl").open("w") as fh:
        for r in lines:
            fh.write(json.dumps(r) + "\n")
    return d


def test_hf_export_layout_and_incremental(tmp_path):
    recs = tmp_path / "recordings"
    _fake_recording(recs, "live-a", "jev")
    _fake_recording(recs, "live-b", "brain")
    (recs / "live-incomplete").mkdir()  # no recording.jsonl: skipped
    out = tmp_path / "hf"

    stats = export(recs, out)
    assert stats["episodes_exported"] == 2
    assert stats["steps_exported"] == 2
    assert stats["frames_linked"] == 2

    # dataset card with HF YAML frontmatter
    card = (out / "README.md").read_text()
    assert card.startswith("---") and "configs:" in card

    episodes = [json.loads(l) for l in
                (out / "data" / "episodes.jsonl").read_text().splitlines()]
    assert {e["episode_id"] for e in episodes} == {"live-a", "live-b"}
    jev_ep = next(e for e in episodes if e["controller"] == "jev")
    assert jev_ep["kills"] == 1 and jev_ep["scenario"] == "fly_arena"

    shards = sorted((out / "data").glob("steps-*.jsonl.gz"))
    assert shards
    steps = [json.loads(l) for l in
             gzip.open(shards[0], "rt").read().splitlines()]
    assert len(steps) == 2
    jev_step = next(s for s in steps if s["episode_id"] == "live-a")
    assert jev_step["selected"] == "attack" and jev_step["attack_spiked"] is True
    assert jev_step["jev"]["intent"] == "engage"
    assert jev_step["frame_file"] == "frames/live-a/000000.jpg"
    assert jev_step["activity_top"] == [[1, 99.0], [2, 50.0]]
    brain_step = next(s for s in steps if s["episode_id"] == "live-b")
    assert brain_step["jev"] is None

    # imagefolder index + hardlinked frames
    meta = [json.loads(l) for l in
            (out / "frames" / "metadata.jsonl").read_text().splitlines()]
    assert len(meta) == 2 and meta[0]["file_name"].endswith("000000.jpg")
    assert (out / "frames" / "live-a" / "000000.jpg").exists()

    # incremental: one new episode appended, no duplicates
    _fake_recording(recs, "live-c", "brain")
    stats2 = export(recs, out)
    assert stats2["episodes_exported"] == 1
    episodes2 = [json.loads(l) for l in
                 (out / "data" / "episodes.jsonl").read_text().splitlines()]
    assert len(episodes2) == 3
    assert len({e["episode_id"] for e in episodes2}) == 3
