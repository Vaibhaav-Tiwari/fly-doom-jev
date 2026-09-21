"""Closed-loop episode runner (v1).

    ViZDoom/fixture -> state/vision -> mock Jev (async) -> JevBridge ->
    LIF engine on reduced MaleCNS -> MotorDecoder -> action -> env

Run:  python -m flydoom.experiments.runner --config configs/demo.yaml
Produces a recording directory under outputs/recordings/<run_id>/.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from flydoom.config import load_config
from flydoom.doom import make_env
from flydoom.integration import JevBridge
from flydoom.jev import JevScheduler, make_jev_client
from flydoom.malecns import load_connectome
from flydoom.malecns.graph import MOTOR_POPULATION
from flydoom.motor import MotorDecoder
from flydoom.neural import LIFEngine
from flydoom.state import encode_state
from flydoom.telemetry import RecordingWriter
from flydoom.vision import RetinaEncoder

log = logging.getLogger(__name__)


def run_episode(cfg: dict, record: bool = True) -> Path | None:
    import flydoom
    seed = int(cfg.get("seed", 42))

    env = make_env(cfg)
    connectome = load_connectome(cfg)
    neural_cfg = cfg["neural"]
    engine = LIFEngine(
        connectome,
        timestep_ms=float(neural_cfg.get("timestep_ms", 2.0)),
        noise=1.0, seed=seed)
    retina = RetinaEncoder(tuple(cfg["vision"].get("retina_size", [16, 12])),
                           gain=float(cfg["vision"].get("gain", 25.0)))
    jev_client = make_jev_client(cfg)
    scheduler = JevScheduler(jev_client, cadence_hz=float(cfg["jev"].get("cadence_hz", 3.0)))
    bridge = JevBridge(connectome, cfg["bridge"]["mappings"],
                       gain=float(cfg["bridge"].get("gain", 30.0)))
    decoder = MotorDecoder(connectome, cfg["motor"]["actions"],
                           threshold=float(cfg["motor"].get("threshold", 0.02)))

    sensory_idx = connectome.population_indices("visual_projection")
    steps_neural = int(neural_cfg.get("steps_per_controller_step", 8))
    dt_neural = float(neural_cfg.get("timestep_ms", 2.0))
    max_steps = int(cfg["environment"].get("max_controller_steps", 175))
    pop_sample = int(cfg["telemetry"].get("population_sample", 32))

    rec_cfg = cfg.get("recording", {})
    writer = None
    if record:
        writer = RecordingWriter(rec_cfg.get("directory", "outputs/recordings"),
                                 record_frames=bool(rec_cfg.get("record_frames", True)))
        writer.write_header({
            "run_kind": "recorded_experiment",
            "software_version": flydoom.__version__,
            "seed": seed,
            "config": cfg,
            "connectome": {"provenance": connectome.provenance,
                           "populations": connectome.population_sizes()},
            "environment_backend": env.backend_name,
            "jev": {"mode": cfg["jev"].get("mode", "mock"), "client": jev_client.name},
            "bridge": bridge.describe(),
            "motor": {"actions": decoder.actions,
                      "source_population": MOTOR_POPULATION},
            "warnings": _fallback_warnings(env, connectome, jev_client),
        })

    scheduler.start()
    stats = {"steps": 0, "kills": 0, "jev_decisions": 0, "jev_errors": 0}
    try:
        obs = env.reset()
        engine.reset()
        scheduler.prime(encode_state(obs))  # synchronous first decision
        t_start = time.time()
        realtime = bool(cfg["environment"].get("realtime", False))
        step_period_s = env.frame_skip / 35.0  # ViZDoom ticrate
        for step_i in range(max_steps):
            t_step = time.perf_counter()
            state = encode_state(obs)
            scheduler.update_state(state)

            decision = scheduler.get_decision()
            # merge sensory drive + Jev modulation into one input write
            s_idx, s_cur = retina.sensory_drive(obs.frame, sensory_idx)
            b_idx, b_cur = bridge.modulation_currents(decision)
            combined: dict[int, float] = {}
            for i, c in zip(s_idx, s_cur):
                combined[int(i)] = combined.get(int(i), 0.0) + float(c)
            for i, c in zip(b_idx, b_cur):
                combined[int(i)] = combined.get(int(i), 0.0) + float(c)
            idx = np.fromiter(combined.keys(), dtype=np.int64)
            cur = np.fromiter(combined.values(), dtype=np.float32)
            engine.inject_input(idx, cur)

            engine.step(steps_neural)
            decoded = decoder.decode(engine.get_motor_output())
            result = env.step(decoded["selected"])
            obs = result.observation

            if writer:
                writer.write_step({
                    "t_ms": round((time.time() - t_start) * 1000.0, 2),
                    "controller_step": step_i,
                    "episode_tic": obs.episode_tic,
                    "state": state.model_dump(),
                    "jev": None if decision is None else {
                        "request_id": decision.request_id,
                        "probabilities": decision.probabilities,
                        "latency_ms": round(decision.latency_ms, 2),
                        "model": decision.model, "is_mock": decision.is_mock,
                        "state_episode_tic": decision.state_episode_tic,
                    },
                    "populations": engine.get_population_activity(pop_sample),
                    "motor": {"scores": decoded["scores"],
                              "selected": decoded["selected"],
                              "confidence": round(decoded["confidence"], 4),
                              "raw_rates": {k: round(v, 3)
                                            for k, v in decoded["raw_rates"].items()}},
                    "reward": result.reward,
                    "controller_latency_ms": round(
                        (time.perf_counter() - t_step) * 1000.0, 3),
                })
                writer.add_frame(obs.frame)
            if realtime:
                elapsed = time.perf_counter() - t_step
                if elapsed < step_period_s:
                    time.sleep(step_period_s - elapsed)
            if result.done:
                break
        stats.update(steps=step_i + 1, kills=obs.kills,
                     jev_decisions=scheduler.decisions_made,
                     jev_errors=scheduler.errors)
    finally:
        scheduler.stop()
        env.close()
        if writer:
            writer.write_step({"kind": "summary", **stats})
            writer.close()

    log.info("episode done: %s", stats)
    if writer:
        print(f"recording: {writer.dir}")
        return writer.dir
    return None


def _fallback_warnings(env, connectome, jev_client) -> list[str]:
    w = []
    if env.backend_name != "vizdoom":
        w.append("ENVIRONMENT FIXTURE: deterministic test env used, NOT real ViZDoom")
    if not connectome.is_real_malecns:
        w.append("CONNECTOME FIXTURE: synthetic test graph used, NOT MaleCNS data")
    if connectome.provenance.get("reduced"):
        w.append("REDUCED CONNECTOME: subset of MaleCNS v1.0, not the full graph")
    if getattr(jev_client, "name", "").startswith("mock"):
        w.append("MOCK JEV: deterministic heuristic, not a live model")
    return w


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a closed-loop demo episode")
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    out = run_episode(cfg, record=not args.no_record)
    if out:
        print(json.dumps({"recording_dir": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
