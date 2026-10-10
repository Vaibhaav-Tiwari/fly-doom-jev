import json

from flydoom.api.replay import list_recordings, load_recording
from flydoom.telemetry import RecordingWriter


def test_recording_roundtrip(tmp_path):
    with RecordingWriter(tmp_path, run_id="test-run", record_frames=True) as w:
        w.write_header({"environment_backend": "fixture"})
        import numpy as np
        ref = w.add_frame(np.full((4, 6, 3), 128, dtype=np.uint8))
        assert ref == "frames/000000.jpg"
        w.write_step({"t_ms": 0.0, "controller_step": 0, "frame_ref": ref,
                      "state": {"health": 100.0}, "jev": None,
                      "populations": {}, "motor": {"selected": "noop"}})

    runs = list_recordings(tmp_path)
    assert [r["run_id"] for r in runs] == ["test-run"]
    rec = load_recording(tmp_path, "test-run")
    assert rec["header"]["environment_backend"] == "fixture"
    assert rec["header"]["recording_format_version"] == "2.2"
    assert len(rec["steps"]) == 1
    assert rec["frames"]["count"] == 1
    assert rec["frames"]["codec"] == "jpeg"
    assert rec["frames"]["url_pattern"].endswith("{:06d}.jpg")
    assert (tmp_path / "test-run" / "frames" / "000000.jpg").exists()


def test_load_missing_recording_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_recording(tmp_path, "does-not-exist")


def test_summary_row_wins_over_episode_end_row(tmp_path):
    recording = tmp_path / "mixed-run"
    recording.mkdir()
    rows = [
        {"kind": "header", "recording_format_version": "2.2"},
        {"kind": "summary", "episode": {"source": "summary"}},
        {"kind": "episode_end", "episode": {"source": "episode_end"}},
    ]
    (recording / "recording.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows)
    )

    loaded = load_recording(tmp_path, "mixed-run")

    # A summary row is canonical, so it must survive alongside a legacy
    # episode_end row rather than being replaced based on file order.
    assert loaded["summary"] == {"episode": {"source": "summary"}}
