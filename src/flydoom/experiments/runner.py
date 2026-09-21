"""Closed-loop episode runner.

    ViZDoom -> state/vision -> Jev (async, mock by default) -> JevBridge ->
    LIF engine on the full MaleCNS v1.0 graph -> typed DN decoder -> action

Run:  python -m flydoom.experiments.runner --config configs/demo.yaml
Produces a recording directory under outputs/recordings/<run_id>/
(recording format v2, see docs/RECORDING_FORMAT.md).
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
from flydoom.malecns.full import FullConnectome
from flydoom.malecns.graph import MOTOR_POPULATION
from flydoom.motor import make_decoder
from flydoom.state import encode_state
from flydoom.telemetry import RecordingWriter
from flydoom.vision import PhotoreceptorPathway, RetinaEncoder

log = logging.getLogger(__name__)

CONNECTOME_ASSET_FILES = {"meta": "meta.json", "positions": "positions.f32",
                          "population": "population.i16", "flags": "flags.u8",
                          "ids": "ids.i64"}


def ensure_connectome_assets(cfg: dict, connectome) -> dict | None:
    """Export the shared browser-viz bundle for the full connectome (once).

    Returns the header reference dict, or None for reduced/fixture graphs
    (small enough that the recording header itself carries positions).
    """
    if not isinstance(connectome, FullConnectome):
        return None
    from flydoom.malecns.viz_export import export_connectome_assets
    out_dir = Path(cfg.get("recording", {}).get("connectome_assets_dir",
                                                "outputs/connectome_assets"))
    if not (out_dir / "meta.json").exists():
        export_connectome_assets(connectome, out_dir)
    return {"url": "/api/connectome", "files": dict(CONNECTOME_ASSET_FILES),
            "key": "neuron_index",
            "note": "arrays are keyed by neuron index (graph row order); "
                    "step activity indices and decoder contributing sets "
                    "reference the same order"}


def _motor_population(connectome) -> tuple[np.ndarray, list[int]]:
    """All descending/motor-population neurons: indices + body ids."""
    if isinstance(connectome, FullConnectome):
        idx = np.sort(np.concatenate([
            connectome.population_indices("descending_neuron"),
            connectome.population_indices("vnc_motor")]))
        ids = connectome.ids
    else:
        idx = connectome.population_indices(MOTOR_POPULATION)
        ids = connectome.body_ids
    return idx, [int(ids[i]) for i in idx]


def _activity_record(engine, motor_idx: np.ndarray, top_k: int) -> dict:
    """Per-step neuron-level activity for the 3D brain view.

    top:    [[neuron_index, rate_hz], ...] for the top_k most active neurons
            (zeros excluded, rates rounded to 0.1 Hz)
    motor_rates: rate_hz for every motor-population neuron, aligned to the
            header's motor_population.indices order
    """
    rate = np.asarray(engine.rate, dtype=np.float64)
    k = int(min(top_k, rate.size))
    if k > 0:
        sel = np.argpartition(-rate, k - 1)[:k]
        sel = sel[np.argsort(-rate[sel])]
        top = [[int(i), round(float(rate[i]), 1)] for i in sel if rate[i] > 0.0]
    else:
        top = []
    motor_rates = [round(float(r), 1) for r in rate[motor_idx]]
    return {"top": top, "motor_rates": motor_rates}


def build_neural(cfg: dict):
    """Returns (connectome, engine). Full native engine by default."""
    connectome = load_connectome(cfg)
    neural = cfg["neural"]
    if isinstance(connectome, FullConnectome):
        from flydoom.neural.engine_native import NativeLIFEngine
        eng = neural.get("engine", {})
        engine = NativeLIFEngine(
            connectome,
            timestep_ms=float(neural.get("timestep_ms", 1.0)),
            syn_gain=float(eng.get("syn_gain", 0.3)),
            tau_v_ms=float(eng.get("tau_v_ms", 20.0)),
            tau_syn_ms=float(eng.get("tau_syn_ms", 5.0)),
            syn_delay_ms=float(eng.get("syn_delay_ms", 2.0)),
            rate_tau_ms=float(eng.get("rate_tau_ms", 100.0)))
    else:
        from flydoom.neural import LIFEngine
        engine = LIFEngine(connectome,
                           timestep_ms=float(neural.get("timestep_ms", 2.0)),
                           noise=1.0, seed=int(cfg.get("seed", 42)))
    return connectome, engine


def build_vision(cfg: dict, connectome):
    if isinstance(connectome, FullConnectome):
        v = cfg["vision"]
        return ("photoreceptor", PhotoreceptorPathway(
            connectome, gain=float(v.get("gain", 30.0)),
            lamina_bias=float(v.get("lamina_bias", 8.0))))
    v = cfg["vision"]
    enc = RetinaEncoder(tuple(v.get("retina_size", [16, 12])),
                        gain=float(v.get("gain", 10.0)))
    return ("mosaic", enc)


def run_episode(cfg: dict, record: bool = True) -> Path | None:
    import flydoom
    seed = int(cfg.get("seed", 42))

    env = make_env(cfg)
    connectome, engine = build_neural(cfg)
    vision_kind, vision = build_vision(cfg, connectome)
    jev_client = make_jev_client(cfg)
    scheduler = JevScheduler(jev_client, cadence_hz=float(cfg["jev"].get("cadence_hz", 3.0)))
    bridge = JevBridge(connectome, cfg["bridge"]["mappings"],
                       gain=float(cfg["bridge"].get("gain", 30.0)))
    decoder = make_decoder(connectome, cfg)

    neural_cfg = cfg["neural"]
    steps_neural = int(neural_cfg.get("steps_per_controller_step", 32))
    dt_neural = float(neural_cfg.get("timestep_ms", 1.0))
    max_steps = int(cfg["environment"].get("max_controller_steps", 175))
    pop_sample = int(cfg["telemetry"].get("population_sample", 32))
    top_k = int(cfg["telemetry"].get("top_k", 256))
    motor_idx, motor_ids = _motor_population(connectome)
    assets_ref = ensure_connectome_assets(cfg, connectome) if record else None

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
                           "n_neurons": connectome.n_neurons,
                           "n_edges": (connectome.n_edges if isinstance(connectome, FullConnectome)
                                       else connectome.weights.nnz),
                           "populations": connectome.population_sizes()},
            "environment": {"backend": env.backend_name,
                            "scenario": getattr(env, "scenario", "fixture"),
                            "actions": env.available_actions},
            "vision": {"pathway": vision_kind},
            "jev": {"mode": cfg["jev"].get("mode", "mock"), "client": jev_client.name,
                    "cadence_hz": float(cfg["jev"].get("cadence_hz", 3.0))},
            "bridge": bridge.describe(),
            "motor": decoder.describe(),
            "motor_population": {"indices": [int(i) for i in motor_idx],
                                 "body_ids": motor_ids},
            "connectome_assets": assets_ref,
            "telemetry": {"population_sample": pop_sample, "top_k": top_k},
            "warnings": _fallback_warnings(env, connectome, jev_client),
        })

    scheduler.start()
    total_reward = 0.0
    action_counts: dict[str, int] = {}
    controller_latencies: list[float] = []
    try:
        obs = env.reset()
        engine.reset()
        scheduler.prime(encode_state(obs, env.available_actions))
        t_start = time.time()
        realtime = bool(cfg["environment"].get("realtime", False))
        step_period_s = env.frame_skip / 35.0  # ViZDoom ticrate
        step_i = 0
        for step_i in range(max_steps):
            t_step = time.perf_counter()
            state = encode_state(obs, env.available_actions)
            scheduler.update_state(state)
            decision = scheduler.get_decision()

            # visual pathway -> sensory drive
            if vision_kind == "photoreceptor":
                s_idx, s_cur = vision.sample(obs.frame, steps_neural * dt_neural)
            else:
                s_idx, s_cur = vision.sensory_drive(
                    obs.frame,
                    connectome.population_indices("visual_projection"))
            # Jev -> bridge modulation (upstream/intermediate populations only)
            b_idx, b_cur = bridge.modulation_currents(decision)
            combined: dict[int, float] = {}
            for i, c in zip(s_idx, s_cur):
                combined[int(i)] = combined.get(int(i), 0.0) + float(c)
            for i, c in zip(b_idx, b_cur):
                combined[int(i)] = combined.get(int(i), 0.0) + float(c)
            engine.inject_input(
                np.fromiter(combined.keys(), dtype=np.int64),
                np.fromiter(combined.values(), dtype=np.float32))

            engine.step(steps_neural)
            decoded = decoder.decode(engine.rate)
            result = env.step(decoded["selected"])
            obs = result.observation
            total_reward += result.reward
            action_counts[decoded["selected"]] = action_counts.get(decoded["selected"], 0) + 1
            latency_ms = (time.perf_counter() - t_step) * 1000.0
            controller_latencies.append(latency_ms)

            if writer:
                frame_ref = writer.add_frame(obs.frame)
                motor_rec = {"scores": decoded["scores"],
                             "selected": decoded["selected"],
                             "confidence": round(float(decoded["confidence"]), 4),
                             "channels": {k: round(v, 4)
                                          for k, v in decoded.get("channels", {}).items()},
                             "readout_rates": decoded.get("readout_rates")}
                writer.write_step({
                    "t_ms": round((time.time() - t_start) * 1000.0, 2),
                    "controller_step": step_i,
                    "episode_tic": obs.episode_tic,
                    "frame_ref": frame_ref,
                    "state": state.model_dump(),
                    "jev": None if decision is None else {
                        "request_id": decision.request_id,
                        "probabilities": decision.probabilities,
                        "latency_ms": round(decision.latency_ms, 2),
                        "model": decision.model, "is_mock": decision.is_mock,
                        "state_episode_tic": decision.state_episode_tic,
                        "usage": decision.meta.get("usage"),
                        "confidence": decision.meta.get("confidence"),
                        "choices": decision.meta.get("choices"),
                    },
                    "neural": {"time_ms": round(engine.time_ms, 2),
                               "steps": steps_neural,
                               "total_spikes": engine.total_spikes},
                    "populations": engine.get_population_activity(pop_sample),
                    "activity": _activity_record(engine, motor_idx, top_k),
                    "motor": motor_rec,
                    "reward": result.reward,
                    "controller_latency_ms": round(latency_ms, 3),
                })
            if realtime:
                elapsed = time.perf_counter() - t_step
                if elapsed < step_period_s:
                    time.sleep(step_period_s - elapsed)
            if result.done:
                break
    finally:
        scheduler.stop()
        env.close()
        if writer:
            metrics = {
                "episode": {
                    "controller_steps": step_i + 1,
                    "duration_s": round(engine.time_ms / 1000.0, 2),
                    "total_reward": round(total_reward, 3),
                    "kills": int(obs.kills),
                    "health_end": float(obs.health),
                    "ammo_end": float(obs.ammo),
                    "action_counts": action_counts,
                    "jev_decisions": scheduler.decisions_made,
                    "jev_errors": scheduler.errors,
                    "controller_latency_ms": {
                        "mean": round(float(np.mean(controller_latencies)), 3),
                        "p95": round(float(np.percentile(controller_latencies, 95)), 3),
                        "max": round(float(np.max(controller_latencies)), 3)}
                    if controller_latencies else {},
                    "neural_total_spikes": engine.total_spikes,
                }}
            writer.write_episode_end(metrics)
            writer.close()

    log.info("episode done: steps=%d kills=%d reward=%.2f jev=%d decisions/%d errors",
             step_i + 1, int(obs.kills), total_reward,
             scheduler.decisions_made, scheduler.errors)
    if writer:
        print(f"recording: {writer.dir}")
        return writer.dir
    return None


def _fallback_warnings(env, connectome, jev_client) -> list[str]:
    w = []
    if env.backend_name != "vizdoom":
        w.append("ENVIRONMENT FIXTURE: deterministic test env used, NOT real ViZDoom")
    src = connectome.provenance.get("source")
    if src == "fixture":
        w.append("CONNECTOME FIXTURE: synthetic test graph used, NOT MaleCNS data")
    elif connectome.provenance.get("reduced"):
        w.append("REDUCED CONNECTOME: subset of MaleCNS v1.0, not the full graph")
    if getattr(jev_client, "name", "").startswith("mock"):
        w.append("MOCK JEV: deterministic heuristic, not a live model")
    return w


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run a closed-loop episode")
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
