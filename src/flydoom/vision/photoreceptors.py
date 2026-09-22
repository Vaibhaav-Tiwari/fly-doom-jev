"""Retinotopic photoreceptor input pathway (full MaleCNS mode).

Each mapped photoreceptor samples the RGB frame at its UV coordinate derived
from its ommatidium column (built in malecns/full.py). R8p receptors sample the
blue channel, R8y green, all others broadband luminance — a display proxy, NOT
calibrated spectral sensitivity (SCIENCE.md). Sampling is bilinear with sRGB
linearization. Drives are Naka-Rushton-compressed and low-pass filtered.

Concepts adapted from DOOMFLY (MIT): doom/game.py:retinal_samples and
doom/native.py drive schedules. Re-implemented here; see PROVENANCE.md.
"""

from __future__ import annotations

import math

import numpy as np

from flydoom.malecns.full import FullConnectome

_LUMA = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


class PhotoreceptorPathway:
    def __init__(self, connectome: FullConnectome, gain: float = 30.0,
                 half_saturation: float = 0.02, lowpass_tau_ms: float = 10.0,
                 lamina_bias: float = 8.0, adaptation_tau_ms: float = 2000.0):
        self.conn = connectome
        self.gain = float(gain)
        self.c = float(half_saturation)
        self.lowpass_tau = float(lowpass_tau_ms)
        self.lamina_bias = float(lamina_bias)
        # Weber-style gain control: divide by a slow running mean of the scene
        # luminance so CONTRAST, not absolute brightness, drives the network
        # (real photoreceptor adaptation; makes the pathway scenario-robust —
        # dark indoor scenes and bright arenas produce comparable drive).
        self.adaptation_tau = float(adaptation_tau_ms)  # 0 disables adaptation
        self.adaptation = 0.2
        self.uv = np.asarray(connectome.retina_uv, dtype=np.float64)
        self.channel = np.asarray(connectome.retina_channel)
        self.luminance = np.zeros(len(connectome.retina), dtype=np.float32)
        self._coords_valid = False

    def _prepare(self, h: int, w: int) -> None:
        x = self.uv[:, 0] * (w - 1)
        y = self.uv[:, 1] * (h - 1)
        self._x0 = np.clip(x.astype(np.int64), 0, w - 1)
        self._y0 = np.clip(y.astype(np.int64), 0, h - 1)
        self._x1 = np.minimum(self._x0 + 1, w - 1)
        self._y1 = np.minimum(self._y0 + 1, h - 1)
        self._dx = (x - self._x0).astype(np.float32)
        self._dy = (y - self._y0).astype(np.float32)
        self._shape = (h, w)
        self._coords_valid = True

    def sample(self, frame_rgb: np.ndarray, interval_ms: float
               ) -> tuple[np.ndarray, np.ndarray]:
        """frame_rgb: (H, W, 3) uint8. Returns (indices, currents) for the engine."""
        if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
            raise ValueError("RGB frame (H,W,3) required")
        h, w = frame_rgb.shape[:2]
        if not self._coords_valid or self._shape != (h, w):
            self._prepare(h, w)
        lin = _linearize(frame_rgb)
        luma = lin @ _LUMA
        # channel map: 0 -> luma, 1 -> green, 2 -> blue
        chans = [luma, lin[:, :, 1], lin[:, :, 2]]

        def gather(img):
            return ((1 - self._dx) * (1 - self._dy) * img[self._y0, self._x0]
                    + self._dx * (1 - self._dy) * img[self._y0, self._x1]
                    + (1 - self._dx) * self._dy * img[self._y1, self._x0]
                    + self._dx * self._dy * img[self._y1, self._x1])

        light = np.where(self.channel == 1, gather(chans[1]),
                         np.where(self.channel == 2, gather(chans[2]),
                                  gather(chans[0])))
        alpha = 1.0 - math.exp(-interval_ms / self.lowpass_tau)
        self.luminance += alpha * (np.clip(light, 0.0, 1.0) - self.luminance)
        signal = self.luminance
        if self.adaptation_tau > 0.0:
            alpha_a = 1.0 - math.exp(-interval_ms / self.adaptation_tau)
            self.adaptation += alpha_a * (float(signal.mean()) - self.adaptation)
            signal = signal / (self.adaptation + 1e-3)
        drive = self.gain * signal / (self.c + signal)
        self.last_drive = drive.astype(np.float32)  # exposed for /state + recordings

        indices = np.concatenate([self.conn.retina, self.conn.lamina])
        currents = np.concatenate(
            [drive.astype(np.float32),
             np.full(len(self.conn.lamina), self.lamina_bias, dtype=np.float32)])
        return indices.astype(np.int64), currents

    def retina_record(self, top_k: int = 256) -> dict:
        """Top-K photoreceptor INPUT DRIVE (mV, Naka-Rushton compressed
        luminance), aligned to graph neuron indices. This is retinal input,
        not spikes — photoreceptors are graded-potential cells in reality and
        our spiking proxy fires rarely by design (SCIENCE.md)."""
        d = getattr(self, "last_drive", None)
        if d is None or len(d) == 0:
            return {"indices": [], "drive": [],
                    "note": "photoreceptor input drive (mV), not spikes"}
        k = int(min(top_k, len(d)))
        sel = np.argpartition(-d, k - 1)[:k]
        sel = sel[np.argsort(-d[sel])]
        return {"indices": [int(self.conn.retina[i]) for i in sel if d[i] > 0.0],
                "drive": [round(float(d[i]), 2) for i in sel if d[i] > 0.0],
                "note": "photoreceptor input drive (mV), not spikes"}

    def reset(self) -> None:
        self.luminance.fill(0.0)
        self.adaptation = 0.2


def _linearize(rgb: np.ndarray) -> np.ndarray:
    p = rgb.astype(np.float32) / 255.0
    return np.where(p <= 0.04045, p / 12.92, ((p + 0.055) / 1.055) ** 2.4)
