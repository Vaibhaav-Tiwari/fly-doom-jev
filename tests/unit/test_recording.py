from flydoom.api.replay import list_recordings, load_recording
from flydoom.telemetry import RecordingWriter

import numpy as np
import pytest
from PIL import Image


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
    with pytest.raises(FileNotFoundError):
        load_recording(tmp_path, "does-not-exist")


def test_float_frame_is_validated_converted_and_written(tmp_path):
    with RecordingWriter(tmp_path, run_id="float-run") as writer:
        ref = writer.add_frame(np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32))
        assert ref == "frames/000000.jpg"

    with Image.open(tmp_path / "float-run" / ref) as image:
        assert image.mode == "RGB"
        assert np.allclose(np.asarray(image)[0, 0], [0, 128, 255], atol=3)


@pytest.mark.parametrize("frame", [
    np.empty((0, 2, 3), dtype=np.uint8),
    np.zeros((2, 3), dtype=np.uint8),
    np.zeros((2, 3, 4), dtype=np.uint8),
    np.zeros((2, 3, 3), dtype=np.int16),
    np.zeros((2, 3, 3), dtype=np.float32),
])
def test_add_frame_rejects_invalid_frames(tmp_path, frame):
    writer = RecordingWriter(tmp_path, run_id="invalid-run")
    if frame.dtype.kind == "f":
        frame[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="RGB frame"):
        writer.add_frame(frame)
    writer.close()


def test_jpeg_quality_is_limited(tmp_path):
    for quality in (0, 96):
        with pytest.raises(ValueError, match="jpeg_quality"):
            RecordingWriter(tmp_path, run_id=f"quality-{quality}", jpeg_quality=quality)


def test_failed_frame_write_is_atomic_and_does_not_advance_counter(tmp_path, monkeypatch):
    writer = RecordingWriter(tmp_path, run_id="atomic-run")
    frame = np.zeros((2, 2, 3), dtype=np.uint8)

    def fail_save(self, *args, **kwargs):
        raise OSError("synthetic JPEG failure")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(OSError, match="synthetic JPEG failure"):
        writer.add_frame(frame)
    assert writer._frame_count == 0
    assert list((tmp_path / "atomic-run" / "frames").iterdir()) == []
    writer.close()


def test_writer_methods_reject_use_after_close(tmp_path):
    writer = RecordingWriter(tmp_path, run_id="closed-run")
    writer.close()
    writer.close()
    for operation in (
        lambda: writer.write_header({}),
        lambda: writer.write_step({}),
        lambda: writer.write_episode_end({}),
        lambda: writer.add_frame(np.zeros((1, 1, 3), dtype=np.uint8)),
    ):
        with pytest.raises(RuntimeError, match="closed"):
            operation()
