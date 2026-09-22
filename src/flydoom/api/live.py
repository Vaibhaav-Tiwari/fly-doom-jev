"""Live mode server: the real closed loop, continuously, over plain HTTP.

A standalone process runs ViZDoom + the full MaleCNS LIF sim + real Jev API
decisions in a worker thread and publishes one JSON snapshot per controller
step. No LLM/agent is involved at runtime — the only external service is the
Jev System One API (credits, not model tokens).

Endpoints (no websockets, simple polling):
  GET  /state  -> latest published snapshot (serialized once per frame)
  POST /new    -> abort current episode, start a fresh one with a new seed
  GET  /health -> {status, uptime_s, episodes, jev: {ok}}

Every episode is structurally recorded with the standard recording format
(recording.jsonl + frames/, format 2.1) under
outputs/recordings/live-<timestamp>-<hash>/ so any live session replays in the
website unchanged. Run:

    set -a; source /Users/vaibhaav/fly-doom-jev/.env; set +a
    python -m flydoom.api.live --config configs/demo.yaml
"""

from __future__ import annotations

import argparse
import base64
import copy
import io
import json
import logging
import os
import sys
import threading
import time

import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

import flydoom
from flydoom.config import load_config
from flydoom.doom import make_env
from flydoom.experiments.runner import (
    _activity_record, _behavior_metrics, _fallback_warnings, _motor_population,
    build_neural, build_vision, ensure_connectome_assets, resolve_neural_steps)
from flydoom.integration import JevBridge
from flydoom.jev import JevScheduler, make_jev_client
from flydoom.motor import make_decoder
from flydoom.state import encode_state
from flydoom.telemetry import RecordingWriter

log = logging.getLogger(__name__)

CORS_ORIGINS = ["http://127.0.0.1:4173", "http://localhost:4173"]


class LiveLoop(threading.Thread):
    """Worker thread: continuous episodes, snapshot publishing, recording."""

    def __init__(self, cfg: dict):
        super().__init__(daemon=True, name="live-loop")
        self.cfg = cfg
        live = cfg.get("live", {})
        self.fps = float(live.get("fps", 6.0))
        self.frame_size = tuple(live.get("frame_size", [640, 480]))  # W, H
        self.jpeg_quality = int(live.get("jpeg_quality", 75))
        self.max_steps = int(cfg["environment"].get("max_controller_steps", 900))
        self.base_seed = int(cfg.get("seed", 42))

        self.status = "starting"
        self.degraded_reason: str | None = None
        self.episodes = 0
        self.started_at = time.time()
        self._reset_requested = threading.Event()
        self._pending_scenario: str | None = None
        self._snapshot_json = json.dumps({"status": "starting"})

    # ------------------------------------------------------------- snapshot
    def snapshot(self) -> str:
        return self._snapshot_json

    def _publish(self, snap: dict) -> None:
        snap["status"] = self.status
        self._snapshot_json = json.dumps(snap, separators=(",", ":"))

    # ------------------------------------------------------------ jev setup
    def _make_jev(self):
        """Live Jev client; degrade to clearly-marked mock if unreachable."""
        if self.cfg["jev"].get("mode") != "live":
            self.degraded_reason = "jev.mode is not 'live' in config (mock client)"
            log.critical("LIVE SERVER DEGRADED: %s", self.degraded_reason)
            self.cfg["jev"]["mode"] = "mock"
            return make_jev_client(self.cfg)
        try:
            client = make_jev_client(self.cfg)
            probe = client.probe()  # GET /models — verifies key + endpoint
            log.info("Jev API reachable: %s", probe)
            return client
        except Exception as exc:
            self.degraded_reason = f"Jev API unreachable at startup: {exc}"
            log.critical("LIVE SERVER DEGRADED: %s — falling back to MOCK Jev",
                         self.degraded_reason)
            self.cfg["jev"]["mode"] = "mock"
            return make_jev_client(self.cfg)

    # ------------------------------------------------------------------ run
    def run(self) -> None:
        cfg = self.cfg
        env = make_env(cfg)
        connectome, engine = build_neural(cfg)
        from flydoom.neural.tonic import maybe_calibrate_tonic
        tonic_meta = maybe_calibrate_tonic(cfg, connectome, engine)
        vision_kind, vision = build_vision(cfg, connectome)
        jev_client = self._make_jev()
        scheduler = JevScheduler(jev_client,
                                 cadence_hz=float(cfg["jev"].get("cadence_hz", 3.0)))
        bridge = JevBridge(connectome, cfg["bridge"]["mappings"],
                           gain=float(cfg["bridge"].get("gain", 30.0)),
                           choice_mappings=cfg["bridge"].get("choice_mappings"))
        decoder = make_decoder(connectome, cfg)
        if tonic_meta.get("enabled"):
            decoder.set_baseline(engine.rate.copy())
        from flydoom.integration.weighting import (apply_action_weighting,
                                                   jev_action_weights)
        weighting = bool(cfg["jev"].get("action_weighting", False))
        steps_neural, dt_neural, ms_per_tic = resolve_neural_steps(cfg)
        pop_sample = int(cfg["telemetry"].get("population_sample", 32))
        top_k = int(cfg["telemetry"].get("top_k", 256))
        motor_idx, motor_ids = _motor_population(connectome)
        assets_ref = ensure_connectome_assets(cfg, connectome)
        rec_cfg = cfg.get("recording", {})
        scenario = cfg["environment"].get("scenario", "fly_arena")
        step_period_s = env.frame_skip / 35.0
        self.status = "degraded_mock_jev" if self.degraded_reason else "running"
        scheduler.start()
        episode_i = 0
        try:
            while True:
                if (self._pending_scenario is not None
                        and self._pending_scenario != scenario):
                    scenario = self._pending_scenario
                    env.close()
                    env_cfg = copy.deepcopy(self.cfg)
                    env_cfg["environment"]["scenario"] = scenario
                    env = make_env(env_cfg)
                    step_period_s = env.frame_skip / 35.0
                    log.info("live scenario switched to %s", scenario)
                self._pending_scenario = None
                episode_i += 1
                seed = self.base_seed + episode_i - 1
                self._run_episode(
                    episode_i=episode_i, seed=seed, env=env, engine=engine,
                    vision_kind=vision_kind, vision=vision, scheduler=scheduler,
                    jev_client=jev_client, bridge=bridge, decoder=decoder,
                    steps_neural=steps_neural, dt_neural=dt_neural,
                    ms_per_tic=ms_per_tic, tonic_meta=tonic_meta,
                    weighting=weighting, apply_weighting=apply_action_weighting,
                    weights_of=jev_action_weights, scenario=scenario,
                    pop_sample=pop_sample, top_k=top_k, motor_idx=motor_idx,
                    motor_ids=motor_ids, assets_ref=assets_ref, rec_cfg=rec_cfg,
                    step_period_s=step_period_s, connectome=connectome)
                self._reset_requested.clear()
        finally:
            scheduler.stop()
            env.close()

    # -------------------------------------------------------------- episode
    def _run_episode(self, episode_i, seed, env, engine, vision_kind, vision,
                     scheduler, jev_client, bridge, decoder, steps_neural,
                     dt_neural, ms_per_tic, tonic_meta, weighting,
                     apply_weighting, weights_of, scenario, pop_sample, top_k,
                     motor_idx, motor_ids, assets_ref, rec_cfg, step_period_s,
                     connectome) -> None:
        import uuid
        run_id = (time.strftime("live-%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
        writer = RecordingWriter(rec_cfg.get("directory", "outputs/recordings"),
                                 run_id=run_id,
                                 record_frames=bool(rec_cfg.get("record_frames", True)))
        writer.write_header({
            "run_kind": "live_session",
            "software_version": flydoom.__version__,
            "seed": seed,
            "config": self.cfg,
            "connectome": {"provenance": connectome.provenance,
                           "n_neurons": connectome.n_neurons,
                           "n_edges": getattr(connectome, "n_edges", None)
                                      or connectome.weights.nnz,
                           "populations": connectome.population_sizes()},
            "environment": {"backend": env.backend_name,
                            "scenario": getattr(env, "scenario", "fixture"),
                            "skill": getattr(env, "skill", None),
                            "actions": env.available_actions},
            "vision": {"pathway": vision_kind},
            "neural": {"timestep_ms": dt_neural,
                       "steps_per_controller_step": steps_neural,
                       "ms_per_game_tic": round(ms_per_tic, 3),
                       "brain_ms_per_game_s": round(ms_per_tic * 35.0, 1),
                       "tonic": tonic_meta,
                       "note": "brain time per game tic; the game runs slower "
                               "in wall time when this is raised"},
            "jev": {"mode": self.cfg["jev"].get("mode", "mock"),
                    "client": jev_client.name,
                    "cadence_hz": float(self.cfg["jev"].get("cadence_hz", 3.0))},
            "bridge": bridge.describe(),
            "motor": decoder.describe(),
            "motor_population": {"indices": [int(i) for i in motor_idx],
                                 "body_ids": motor_ids},
            "connectome_assets": assets_ref,
            "telemetry": {"population_sample": pop_sample, "top_k": top_k},
            "warnings": (_fallback_warnings(env, connectome, jev_client)
                         + ([f"DEGRADED LIVE MODE: {self.degraded_reason}"]
                            if self.degraded_reason else [])),
        })

        if hasattr(env, "set_seed"):
            env.set_seed(seed)
        obs = env.reset()
        engine.reset()
        # no blocking prime(): the scheduler's background loop produces the
        # first decision within ~1s; until then the stale-decision semantics
        # keep the previous decision active (recorded honestly per step)
        scheduler.update_state(encode_state(obs, env.available_actions))
        self.episodes += 1
        log.info("live episode %d started: run_id=%s seed=%d",
                 episode_i, run_id, seed)

        t_start = time.time()
        jev_d0 = scheduler.decisions_made
        jev_e0 = scheduler.errors
        sequence = 0
        last_frame_b64: str | None = None
        last_frame_t = 0.0
        total_reward = 0.0
        latencies: list[float] = []
        beh: dict[str, list] = {"turn_imb": [], "selected": [], "visible": [],
                                "dist": [], "angle": [], "attack_ch": []}
        reset_reason = "episode_finished"
        step_i = -1
        for step_i in range(self.max_steps):
            if self._reset_requested.is_set():
                reset_reason = "manual_reset"
                break
            t_step = time.perf_counter()
            state = encode_state(obs, env.available_actions)
            scheduler.update_state(state)
            decision = scheduler.get_decision()

            if vision_kind == "photoreceptor":
                s_idx, s_cur = vision.sample(obs.frame, steps_neural * dt_neural)
                retina_rec = vision.retina_record(top_k)
            else:
                s_idx, s_cur = vision.sensory_drive(
                    obs.frame, connectome.population_indices("visual_projection"))
                retina_rec = None
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
            if weighting:
                decoded = apply_weighting(
                    decoded, weights_of(list(decoded["scores"]), decision))
            combo = decoded.get("combo") or [decoded["selected"]]
            result = env.step(combo)
            obs = result.observation
            total_reward += result.reward
            latencies.append((time.perf_counter() - t_step) * 1000.0)
            beh["turn_imb"].append(float(decoded.get("imbalances", {}).get("turn", 0.0)))
            beh["attack_ch"].append(float(decoded.get("channels", {}).get("attack", 0.0)))
            beh["selected"].append(decoded["selected"])
            beh["visible"].append(bool(state.enemy_visible))
            beh["dist"].append(float(state.enemy_distance))
            beh["angle"].append(float(state.enemy_angle))

            frame_ref = writer.add_frame(obs.frame)
            sequence += 1
            now = time.time()
            if now - last_frame_t >= 1.0 / self.fps or last_frame_b64 is None:
                last_frame_b64 = self._encode_frame(obs.frame)
                last_frame_t = now
            activity = _activity_record(engine, motor_idx, top_k)
            pops = {g: round(v["mean_rate_hz"], 3)
                    for g, v in engine.get_population_activity(None).items()}
            self._publish({
                "run_id": run_id,
                "scenario": scenario,
                "sequence": sequence,
                "generated_at_ms": round(now * 1000.0, 1),
                "frame": "data:image/jpeg;base64," + last_frame_b64,
                "game": {"health": float(obs.health), "kills": int(obs.kills),
                         "ammo": float(obs.ammo),
                         "enemies": (env.live_enemy_count()
                                     if hasattr(env, "live_enemy_count") else None),
                         "alive_s": round(obs.episode_tic / 35.0, 2),
                         "episode": episode_i},
                "motor": {"selected": decoded["selected"],
                          "combo": combo,
                          "scores": {k: round(float(v), 4)
                                     for k, v in decoded["scores"].items()},
                          "neural_scores": ({k: round(float(v), 4)
                                             for k, v in decoded["neural_scores"].items()}
                                            if decoded.get("neural_scores") else None),
                          "jev_weights": decoded.get("jev_weights"),
                          "channels": {k: round(float(v), 4)
                                       for k, v in decoded.get("channels", {}).items()},
                          "readout_rates": decoded.get("readout_rates")},
                "retina": retina_rec,
                "activity": activity,
                "populations": pops,
                "jev": None if decision is None else {
                    "model": decision.model, "is_mock": decision.is_mock,
                    "probabilities": decision.probabilities,
                    "latency_ms": round(decision.latency_ms, 2),
                    "usage": decision.meta.get("usage"),
                    "choices": decision.meta.get("choices"),
                    "choice_probabilities": decision.meta.get("choice_probabilities"),
                },
            })

            writer.write_step({
                "t_ms": round((now - t_start) * 1000.0, 2),
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
                    "choice_probabilities": decision.meta.get("choice_probabilities"),
                },
                "neural": {"time_ms": round(engine.time_ms, 2),
                           "steps": steps_neural,
                           "total_spikes": engine.total_spikes},
                "retina": retina_rec,
                "populations": engine.get_population_activity(pop_sample),
                "activity": activity,
                "motor": {"scores": decoded["scores"],
                          "selected": decoded["selected"],
                          "combo": combo,
                          "confidence": round(float(decoded["confidence"]), 4),
                          "neural_scores": decoded.get("neural_scores"),
                          "jev_weights": decoded.get("jev_weights"),
                          "channels": {k: round(v, 4)
                                       for k, v in decoded.get("channels", {}).items()},
                          "imbalances": decoded.get("imbalances"),
                          "readout_rates": decoded.get("readout_rates")},
                "reward": result.reward,
                "controller_latency_ms": round(latencies[-1], 3),
            })
            if result.done:
                break
            elapsed = time.perf_counter() - t_step
            if elapsed < step_period_s:
                time.sleep(step_period_s - elapsed)

        episode = {
            "controller_steps": step_i + 1,
            "episode_tics": int(obs.episode_tic),
            "survival_s": round(obs.episode_tic / 35.0, 2),
            "duration_s": round(engine.time_ms / 1000.0, 2),
            "total_reward": round(total_reward, 3),
            "kills": int(obs.kills),
            "health_end": float(obs.health),
            "ammo_end": float(obs.ammo),
            "reset_reason": reset_reason,
            "jev_decisions": scheduler.decisions_made - jev_d0,
            "jev_errors": scheduler.errors - jev_e0,
            "jev_disabled_reason": scheduler.disabled_reason,
            "controller_latency_ms": {
                "mean": round(float(np.mean(latencies)), 3),
                "p95": round(float(np.percentile(latencies, 95)), 3),
                "max": round(float(np.max(latencies)), 3)}
            if latencies else {},
            "neural_total_spikes": engine.total_spikes,
            "behavior": _behavior_metrics(beh),
        }
        writer.write_episode_end({"episode": episode})
        writer.close()
        log.info("live episode %d done: run_id=%s tics=%d kills=%d reason=%s",
                 episode_i, run_id, int(obs.episode_tic), int(obs.kills),
                 reset_reason)

    # ----------------------------------------------------------------- jpeg
    def _encode_frame(self, frame_rgb: np.ndarray) -> str:
        from PIL import Image
        img = Image.fromarray(np.ascontiguousarray(frame_rgb))
        if img.size != self.frame_size:
            img = img.resize(self.frame_size, Image.NEAREST)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=self.jpeg_quality)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # ----------------------------------------------------------------- api
    def request_new_episode(self, scenario: str | None = None) -> str | None:
        """Abort the current episode; optionally switch scenario next episode.
        Returns an error string for an unknown scenario, else None."""
        if scenario is not None:
            from flydoom.doom.base import SCENARIOS
            if scenario not in SCENARIOS:
                return (f"unknown scenario {scenario!r}; "
                        f"supported: {sorted(SCENARIOS)}")
            self._pending_scenario = scenario
        self._reset_requested.set()
        return None

    def health(self) -> dict:
        jev_ok = self.degraded_reason is None
        return {"status": self.status,
                "uptime_s": round(time.time() - self.started_at, 1),
                "episodes": self.episodes,
                "jev": {"ok": jev_ok,
                        "degraded_reason": self.degraded_reason}}


def create_app(cfg: dict) -> tuple:
    loop = LiveLoop(cfg)
    app = FastAPI(title="jev-doom-fly live", version=flydoom.__version__)
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS,
                       allow_methods=["GET", "POST"], allow_headers=["*"])

    @app.get("/state")
    def state() -> Response:
        return Response(content=loop.snapshot(), media_type="application/json")

    @app.post("/new")
    def new(body: dict | None = None) -> dict:
        """Start a fresh episode (new seed). Optional body {"scenario": name}
        switches the environment for the next episode."""
        scenario = (body or {}).get("scenario")
        err = loop.request_new_episode(scenario=scenario)
        if err:
            return {"status": "error", "error": err}
        return {"status": "reset_requested",
                "scenario": scenario or "unchanged"}

    @app.get("/health")
    def health() -> dict:
        return loop.health()

    loop.start()
    return app, loop


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Live mode server (real loop, HTTP polling)")
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)
    host = args.host or cfg.get("api", {}).get("host", "127.0.0.1")
    port = args.port or int(cfg.get("api", {}).get("port", 8420))
    app, loop = create_app(cfg)

    import uvicorn
    print(f"live server: http://{host}:{port}  (GET /state, POST /new, GET /health)")
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
