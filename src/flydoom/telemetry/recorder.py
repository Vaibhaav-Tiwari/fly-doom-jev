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
import time
import uuid
from pathlib import Path

import numpy as np

RECORDING_FORMAT_VERSION = "2.1"


class RecordingWriter:
    def __init__(self, directory: str | Path, run_id: str | None = None,
                 record_frames: bool = True, jpeg_quality: int = 80):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.dir = Path(directory) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.dir / "recording.jsonl", "w", encoding="utf-8")
        self.record_frames = record_frames
        self.jpeg_quality = int(jpeg_quality)
        self._frame_count = 0
        self._closed = False
        if record_frames:
            (self.dir / "frames").mkdir(exist_ok=True)

    def write_header(self, meta: dict) -> None:
        self._write({"kind": "header",
                     "recording_format_version": RECORDING_FORMAT_VERSION,
                     "created_at": time.time(), **meta})

    def write_step(self, record: dict) -> None:
        self._write({"kind": "step", **record})

    def write_episode_end(self, metrics: dict) -> None:
        self._write({"kind": "episode_end", "ended_at": time.time(), **metrics})

    def add_frame(self, frame_rgb: np.ndarray) -> str | None:
        """Store an RGB frame as JPEG; returns the frame_ref (relative path)."""
        if not self.record_frames:
            return None
        from PIL import Image
        name = f"frames/{self._frame_count:06d}.jpg"
        img = frame_rgb
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(img).save(self.dir / name, quality=self.jpeg_quality)
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
