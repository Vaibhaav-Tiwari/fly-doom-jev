"""Episode-start settle: mute the power-on transient, calibrate turn zero.

Owner measurement (2026-09-22, 3 live recordings): the first controller step
of EVERY episode decodes turn = +38.57 (identical — the deterministic
connectome power-on surge when the first frame hits the retina), decaying to
a persistent rightward bias (+0.3 -> +0.05, also identical). Two remedies,
both cheap and recorded honestly:

1. SETTLE PERIOD. After env.reset() + engine.reset(), run the brain on the
   spawn frame for `motor.settle_ms` of neural time WITHOUT sending actions
   to the game (motor muted — the game clock does not advance), so the onset
   transient decays before the fly can move.
2. TURN ZERO CALIBRATION. The mean raw turn imbalance over the settled half
   of the window becomes the decoder's turn_offset (subtracted in decode), so
   a symmetric spawn view yields ~0 instead of a standing rightward bias.
   The calibration is stored in the episode metrics and /state.

Engineering housekeeping, not biology (like the tonic baseline).
"""

from __future__ import annotations

import numpy as np


def settle_episode_start(engine, vision, vision_kind: str, frame, decoder,
                         connectome, steps_per_iter: int, dt_ms: float,
                         settle_ms: float) -> dict:
    """Run the settle window; returns the calibration record (for telemetry)."""
    if settle_ms <= 0:
        return {"enabled": False}
    iters = max(1, round(settle_ms / (steps_per_iter * dt_ms)))
    imbs: list[float] = []
    for k in range(iters):
        if vision_kind == "photoreceptor":
            s_idx, s_cur = vision.sample(frame, steps_per_iter * dt_ms)
        else:
            s_idx, s_cur = vision.sensory_drive(
                frame, connectome.population_indices("visual_projection"))
        engine.inject_input(s_idx, s_cur)
        engine.step(steps_per_iter)
        if k >= iters // 2:  # measure only the settled half
            imbs.append(decoder.turn_imbalance(engine.rate))
    offset = float(np.mean(imbs)) if imbs else 0.0
    decoder.set_turn_offset(offset)
    return {"enabled": True,
            "settle_ms": float(settle_ms),
            "iterations": iters,
            "motor_muted": True,
            "turn_offset": round(offset, 4),
            "raw_imbalances": [round(v, 4) for v in imbs],
            "note": "brain ran on the spawn frame with motor output muted; "
                    "mean settled turn imbalance is subtracted as turn zero"}
