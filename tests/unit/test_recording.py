from flydoom.api.replay import list_recordings, load_recording
from flydoom.telemetry import RecordingWriter


def test_recording_roundtrip(tmp_path):
    with RecordingWriter(tmp_path, run_id="test-run", record_frames=True) as w:
        w.write_header({"environment_backend": "fixture"})
        import numpy as np
        w.add_frame(np.ones((4, 6), dtype=np.float32))
        w.write_step({"t_ms": 0.0, "controller_step": 0,
                      "state": {"health": 100.0}, "jev": None,
                      "populations": {}, "motor": {"selected": "noop"}})

    runs = list_recordings(tmp_path)
    assert [r["run_id"] for r in runs] == ["test-run"]
    rec = load_recording(tmp_path, "test-run")
    assert rec["header"]["environment_backend"] == "fixture"
    assert len(rec["steps"]) == 1
    assert rec["frames"]["count"] == 1
    assert (tmp_path / "test-run" / "frames.u8").exists()


def test_load_missing_recording_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_recording(tmp_path, "does-not-exist")
