"""Recording writer — RECORDING FORMAT v2.

One recording = one run directory:

  recording.jsonl   line-delimited records; first line is the header
                    (kind=header), then per-controller-step records
                    (kind=step), then episode summaries (kind=episode_end)
                    and a footer (kind=footer). See docs/RECORDING_FORMAT.md.
  frames/NNNNNN.jpg RGB frames as JPEG (when record_frames is on)

Frames are referenced from step records via `frame_ref`. Everything the
dashboard needs is under the run directory; no external services are required
for replay.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np

RECORDING_FORMAT_VERSION = "2.2"


class RecordingWriter:
    def __init__(self, directory: str | Path, run_id: str | None = None,
                 record_frames: bool = True, jpeg_quality: int = 80):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.record_frames = record_frames
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, (int, np.integer)):
            raise ValueError("jpeg_quality must be an integer from 1 through 95")
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 95:
            raise ValueError("jpeg_quality must be an integer from 1 through 95")
        self.dir = Path(directory) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.dir / "recording.jsonl", "w", encoding="utf-8")
        self._frame_count = 0
        self._closed = False
        if record_frames:
            (self.dir / "frames").mkdir(exist_ok=True)

    def write_header(self, meta: dict) -> None:
        self._ensure_open()
        self._write({"kind": "header",
                     "recording_format_version": RECORDING_FORMAT_VERSION,
                     "created_at": time.time(), **meta})

    def write_step(self, record: dict) -> None:
        self._ensure_open()
        self._write({"kind": "step", **record})

    def write_episode_end(self, metrics: dict) -> None:
        self._ensure_open()
        self._write({"kind": "episode_end", "ended_at": time.time(), **metrics})

    def add_frame(self, frame_rgb: np.ndarray) -> str | None:
        """Store an RGB frame as JPEG; returns the frame_ref (relative path)."""
        self._ensure_open()
        if not self.record_frames:
            return None
        from PIL import Image

        if not isinstance(frame_rgb, np.ndarray):
            raise ValueError("RGB frame must be a nonempty HxWx3 numpy array")
        if frame_rgb.ndim != 3 or frame_rgb.shape[0] == 0 or frame_rgb.shape[1] == 0 or frame_rgb.shape[2] != 3:
            raise ValueError("RGB frame must be a nonempty HxWx3 numpy array")
        if frame_rgb.dtype == np.uint8:
            img = frame_rgb
        elif np.issubdtype(frame_rgb.dtype, np.floating):
            if not np.isfinite(frame_rgb).all():
                raise ValueError("RGB frame float values must be finite")
            img = (np.clip(frame_rgb, 0, 1) * 255).astype(np.uint8)
        else:
            raise ValueError("RGB frame must have dtype uint8 or a floating-point dtype")

        name = f"frames/{self._frame_count:06d}.jpg"
        destination = self.dir / name
        fd, temporary = tempfile.mkstemp(
            prefix=f".{destination.stem}-", suffix=destination.suffix, dir=destination.parent
        )
        os.close(fd)
        try:
            Image.fromarray(img, mode="RGB").save(temporary, quality=self.jpeg_quality)
            os.replace(temporary, destination)
        except Exception:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        self._frame_count += 1
        return name

    def close(self) -> None:
        if self._closed:
            return
        if self.record_frames:
            self._write({"kind": "frames_meta", "pattern": "frames/{:06d}.jpg",
                         "count": self._frame_count, "codec": "jpeg",
                         "quality": self.jpeg_quality})
        self._write({"kind": "footer", "ended_at": time.time()})
        self._fp.close()
        self._closed = True

    def _write(self, obj: dict) -> None:
        self._fp.write(json.dumps(obj, default=_json_default) + "\n")
        self._fp.flush()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("recording writer is closed")

    def __enter__(self) -> "RecordingWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not serializable: {type(o)}")
