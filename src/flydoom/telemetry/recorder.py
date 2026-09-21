"""JSONL episode recorder.

One recording = one run directory containing:
  recording.jsonl  line 1: header (provenance/config/versions)
                   then one {"kind": "step", ...} per controller step
                   last lines: frames_meta, footer
  frames.u8        raw uint8 grayscale frames (downsampled), if enabled

Everything needed for browser replay lives under the run directory.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np

RECORDING_FORMAT_VERSION = "1.0"


class RecordingWriter:
    def __init__(self, directory: str | Path, run_id: str | None = None,
                 record_frames: bool = True, frame_downsample: int = 2):
        self.run_id = run_id or time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
        self.dir = Path(directory) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._fp = open(self.dir / "recording.jsonl", "w", encoding="utf-8")
        self.record_frames = record_frames
        self.frame_downsample = int(frame_downsample)
        self._frames_fp = open(self.dir / "frames.u8", "wb") if record_frames else None
        self._frame_count = 0
        self._frame_shape: list[int] | None = None
        self._closed = False

    def write_header(self, meta: dict) -> None:
        self._write({"kind": "header",
                     "recording_format_version": RECORDING_FORMAT_VERSION,
                     "created_at": time.time(), **meta})

    def write_step(self, record: dict) -> None:
        self._write({"kind": "step", **record})

    def add_frame(self, frame: np.ndarray) -> None:
        if self._frames_fp is None:
            return
        ds = self.frame_downsample
        small = np.ascontiguousarray((frame[::ds, ::ds] * 255).astype(np.uint8))
        self._frames_fp.write(small.tobytes())
        self._frame_count += 1
        self._frame_shape = list(small.shape)

    def close(self) -> None:
        if self._closed:
            return
        if self._frames_fp is not None:
            self._frames_fp.close()
            self._write({"kind": "frames_meta", "file": "frames.u8",
                         "count": self._frame_count, "shape": self._frame_shape,
                         "dtype": "uint8", "downsample": self.frame_downsample})
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
