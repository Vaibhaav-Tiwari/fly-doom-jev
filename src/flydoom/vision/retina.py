"""Deterministic visual pathway (v1).

Frame -> mean-pooled grayscale retina (16x12) -> topographically tiled drive of
the ``visual_projection`` sensory population. Where biology ends and engineering
begins: MaleCNS visual_projection neurons are real connectome neurons, but the
retina->neuron assignment is a fixed round-robin tiling, NOT a mapped
retinotopic projection. Documented in SCIENCE.md.
"""

from __future__ import annotations

import numpy as np


class RetinaEncoder:
    def __init__(self, retina_size: tuple[int, int] = (16, 12), gain: float = 25.0):
        self.cols, self.rows = retina_size
        self.gain = float(gain)
        self._assignments: np.ndarray | None = None

    def encode_frame(self, frame: np.ndarray) -> np.ndarray:
        """Mean-pool a frame into a (rows*cols,) retinal vector in [0,1].

        Accepts float (H, W) in [0,1] or uint8 RGB (H, W, 3)."""
        if frame.ndim == 3:
            frame = (frame.astype(np.float32) @ np.asarray(
                [0.2126, 0.7152, 0.0722], dtype=np.float32)) / 255.0
        h, w = frame.shape
        out = np.empty((self.rows, self.cols), dtype=np.float32)
        for r in range(self.rows):
            y0, y1 = r * h // self.rows, (r + 1) * h // self.rows
            for c in range(self.cols):
                x0, x1 = c * w // self.cols, (c + 1) * w // self.cols
                out[r, c] = frame[y0:y1, x0:x1].mean()
        return out.ravel()

    def sensory_drive(self, frame: np.ndarray, sensory_indices: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray]:
        """Return (indices, currents) driving the sensory population from a frame.

        The retinal vector is tiled across the sensory population round-robin,
        so each neuron receives a deterministic spatially-meaningful input.
        """
        retina = self.encode_frame(frame)
        n = len(sensory_indices)
        currents = (retina[np.arange(n) % len(retina)] * self.gain).astype(np.float32)
        return np.asarray(sensory_indices, dtype=np.int64), currents
