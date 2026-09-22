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
import math
import os
import sys
import threading
import time
from pathlib import Path

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

# Public demo: the Vercel-hosted site (and its preview deploys) poll this
# server from arbitrary origins. No credentials are used, so "*" is safe;
# POST /new is intentionally public — visitors may start a fresh game.
CORS_ORIGINS = ["*"]


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
        self.controller = cfg.get("controller", "jev")  # 'jev' | 'brain'
        self._reset_requested = threading.Event()
        self._reset_learning = False  # POST /new {reset_learning:true}
        self._pending_scenario: str | None = None
        self._pending_controller: str | None = None
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
        jev_cfg = cfg["jev"]
        scheduler = JevScheduler(
            jev_client, cadence_hz=float(jev_cfg.get("cadence_hz", 3.0)),
            fast_questions=tuple(jev_cfg.get("fast_questions", ["ATTACK"])),
            strategy_period_s=jev_cfg.get("strategy_period_s"))
        # Persistent strategy memory (owner-directed "train Jev's behavior"):
        # episode outcomes survive restarts so Jev's prompt carries long-term
        # lessons, not just this process's last 3 episodes.
        try:
            mem = Path("outputs/learning/jev_memory.jsonl")
            if mem.exists():
                scheduler.episode_history = [
                    json.loads(l)["summary"]
                    for l in mem.read_text().splitlines()[-3:] if l.strip()]
        except (OSError, json.JSONDecodeError, KeyError):
            pass
        bridge = JevBridge(connectome, cfg["bridge"]["mappings"],
                           gain=float(cfg["bridge"].get("gain", 30.0)),
                           choice_mappings=cfg["bridge"].get("choice_mappings"))
        from flydoom.config import apply_scenario_profile
        scen_cfg = apply_scenario_profile(
            cfg, cfg["environment"].get("scenario", "fly_arena"))
        decoder = make_decoder(connectome, scen_cfg)  # profiled motor settings
        if tonic_meta.get("enabled"):
            decoder.set_baseline(engine.rate.copy())
        dec_baseline = decoder.baseline  # re-applied when the decoder is
                                         # rebuilt on a scenario switch
        from flydoom.integration.weighting import (apply_action_weighting,
                                                   jev_action_weights)
        weighting = bool(cfg["jev"].get("action_weighting", False))
        intent_biases = cfg["jev"].get("intent_biases")

        def weights_of(scores, decision, aim_ok=None):
            return jev_action_weights(scores, decision,
                                      intent_biases=intent_biases,
                                      aim_ok=aim_ok)
        from flydoom.neural.plasticity import (maybe_make_plasticity,
                                               plasticity_config_sha,
                                               shaped_reward)
        plasticity = maybe_make_plasticity(cfg, connectome, engine)
        self._checkpoint_path = None
        self._cfg_sha = None
        self._last_checkpoint_t = time.time()
        if plasticity:
            self._cfg_sha = plasticity_config_sha(cfg)
            self._checkpoint_path = Path(cfg.get("plasticity", {}).get(
                "checkpoint", "outputs/learning/checkpoint.npz"))
            # one continuously-learning fly: reload learned weights across
            # server restarts when provenance (graph + config hash) matches
            plasticity.load_checkpoint(self._checkpoint_path, self._cfg_sha)
        steps_neural, dt_neural, ms_per_tic = resolve_neural_steps(cfg)
        pop_sample = int(cfg["telemetry"].get("population_sample", 32))
        top_k = int(cfg["telemetry"].get("top_k", 256))
        motor_idx, motor_ids = _motor_population(connectome)
        assets_ref = ensure_connectome_assets(cfg, connectome)
        rec_cfg = cfg.get("recording", {})
        scenario = cfg["environment"].get("scenario", "fly_arena")
        step_period_s = env.frame_skip / 35.0
        scheduler.set_active(self.controller == "jev")  # brain mode: no calls
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
                    # scenario-aware control profile: rebuild the decoder with
                    # the profile-merged motor settings (aim gate, forward_min)
                    scen_cfg = apply_scenario_profile(self.cfg, scenario)
                    decoder = make_decoder(connectome, scen_cfg)
                    if dec_baseline is not None:
                        decoder.set_baseline(dec_baseline)
                    log.info("live scenario switched to %s", scenario)
                self._pending_scenario = None
                if (self._pending_controller is not None
                        and self._pending_controller != self.controller):
                    self.controller = self._pending_controller
                    # brain-only mode: Jev fully bypassed — scheduler paused,
                    # ZERO API calls; decoder readout alone drives actions
                    scheduler.set_active(self.controller == "jev")
                    log.info("live controller switched to %s", self.controller)
                self._pending_controller = None
                episode_i += 1
                seed = self.base_seed + episode_i - 1
                reason = self._run_episode(
                    episode_i=episode_i, seed=seed, env=env, engine=engine,
                    vision_kind=vision_kind, vision=vision, scheduler=scheduler,
                    jev_client=jev_client, bridge=bridge, decoder=decoder,
                    steps_neural=steps_neural, dt_neural=dt_neural,
                    ms_per_tic=ms_per_tic, tonic_meta=tonic_meta,
                    weighting=weighting, apply_weighting=apply_action_weighting,
                    weights_of=weights_of, scenario=scenario,
                    pop_sample=pop_sample, top_k=top_k, motor_idx=motor_idx,
                    motor_ids=motor_ids, assets_ref=assets_ref, rec_cfg=rec_cfg,
                    step_period_s=step_period_s, connectome=connectome,
                    plasticity=plasticity, shaped_reward=shaped_reward,
                    controller=self.controller, scen_cfg=scen_cfg)
                # learned weights persist across natural episode ends AND
                # manual resets (one continuously-learning fly); only an
                # explicit POST /new {reset_learning:true} starts the fly fresh
                if plasticity and reason == "manual_reset" \
                        and self._reset_learning:
                    plasticity.reset()
                    if self._checkpoint_path and self._checkpoint_path.exists():
                        self._checkpoint_path.unlink()  # lineage restarts
                self._reset_learning = False
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
                     connectome, plasticity, shaped_reward, controller,
                     scen_cfg) -> str:
        import uuid
        motor_cfg = scen_cfg.get("motor", {})
        aim_cone = float(motor_cfg.get("attack_aim_cone_deg", 14.0))
        aim_range = float(motor_cfg.get("attack_range", 0.45))
        unstuck_cfg = scen_cfg.get("unstuck", {})
        unstuck = None
        if unstuck_cfg.get("enabled"):
            from flydoom.motor.reflexes import UnstuckReflex
            unstuck = UnstuckReflex(
                window_s=float(unstuck_cfg.get("window_s", 2.5)),
                epsilon=float(unstuck_cfg.get("epsilon", 6.0)),
                turn_s=float(unstuck_cfg.get("turn_s", 1.0)))
        assist = None
        assist_hold = int(motor_cfg.get("aim_assist_hold_steps", 2))
        if assist_hold > 0:
            from flydoom.motor.reflexes import AimAssistReflex
            assist = AimAssistReflex(hold_steps=assist_hold)
        # Level goals (owner-directed): fly_arena is CLEARED when all enemies
        # are dead (no wandering after the last kill); e1m1 is CLEARED by
        # reaching the real exit linedef (geometry from e1m1_map.json), and
        # the exit bearing/distance feeds both /state (minimap) and Jev.
        goal = None
        if scenario == "e1m1":
            mp = Path(__file__).parents[1] / "scenarios" / "e1m1_map.json"
            if mp.exists():
                _m = json.loads(mp.read_text())
                if _m.get("exit"):
                    goal = (float(_m["exit"][0]), float(_m["exit"][1]))
        arena_enemies = int(scen_cfg.get("enemy_count", 3))
        run_id = (time.strftime("live-%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6])
        writer = RecordingWriter(rec_cfg.get("directory", "outputs/recordings"),
                                 run_id=run_id,
                                 record_frames=bool(rec_cfg.get("record_frames", True)))
        writer.write_header({
            "run_kind": "live_session",
            "software_version": flydoom.__version__,
            "seed": seed,
            "controller": controller,  # 'jev' (Jev+MaleCNS) | 'brain' (MaleCNS only)
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
            "scenario_profile": {"scenario": scenario,
                                 "motor": {"forward_min": motor_cfg.get("forward_min"),
                                           "attack_aim_cone_deg": aim_cone,
                                           "attack_range": aim_range},
                                 "unstuck": unstuck_cfg or None},
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
            "plasticity": (({**plasticity.describe(),
                             "checkpoint_id": plasticity.checkpoint_id,
                             "checkpoint": str(self._checkpoint_path)})
                           if plasticity else {"enabled": False}),
            "motor_population": {"indices": [int(i) for i in motor_idx],
                                 "body_ids": motor_ids},
            "connectome_assets": assets_ref,
            "telemetry": {"population_sample": pop_sample, "top_k": top_k},
            "warnings": (_fallback_warnings(env, connectome, jev_client)
                         + (["BRAIN-ONLY CONTROLLER: Jev bypassed by design "
                             "(zero API calls); actions are pure neural "
                             "decoder readout"] if controller == "brain" else [])
                         + ([f"DEGRADED LIVE MODE: {self.degraded_reason}"]
                            if self.degraded_reason else [])),
        })

        if hasattr(env, "set_seed"):
            env.set_seed(seed)
        obs = env.reset()
        engine.reset()
        from flydoom.neural.settle import settle_episode_start
        settle_meta = settle_episode_start(
            engine, vision, vision_kind, obs.frame, decoder, connectome,
            steps_neural, dt_neural,
            float(motor_cfg.get("settle_ms", 0.0)))
        # no blocking prime(): the scheduler's background loop produces the
        # first decision within ~1s; until then the stale-decision semantics
        # keep the previous decision active (recorded honestly per step)
        scheduler.update_state(encode_state(obs, env.available_actions))
        if controller == "brain":
            log.info("live episode %d: BRAIN-ONLY controller — Jev bypassed, "
                     "zero API calls this episode", episode_i)
        self.episodes += 1
        log.info("live episode %d started: run_id=%s seed=%d",
                 episode_i, run_id, seed)

        t_start = time.time()
        prev_obs = obs
        from collections import deque
        dmg_window = max(2, round(35.0 / env.frame_skip))  # ~1 s of game time
        health_hist: deque = deque(maxlen=dmg_window + 1)
        health_hist.append(float(obs.health))
        jev_d0 = scheduler.decisions_made
        jev_e0 = scheduler.errors
        sequence = 0
        last_frame_b64: str | None = None
        last_frame_t = 0.0
        total_reward = 0.0
        reward_terms: dict[str, int] = {}
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
            state = encode_state(obs, env.available_actions,
                                 recent_damage=max(0.0, health_hist[0]
                                                   - float(obs.health)))
            goal_pub = None
            if goal is not None:
                gdx, gdy = goal[0] - obs.position_x, goal[1] - obs.position_y
                goal_dist = math.hypot(gdx, gdy)
                goal_pub = {"x": goal[0], "y": goal[1],
                            "dist": round(goal_dist, 1),
                            "bearing_deg": round(math.degrees(
                                math.atan2(gdy, gdx)), 1)}
                state["goal"] = {"exit_dist_units": round(goal_dist, 1),
                                 "exit_bearing_deg": goal_pub["bearing_deg"],
                                 "note": "reach the exit lift to clear E1M1"}
            if controller == "jev":
                scheduler.update_state(state)
                decision = scheduler.get_decision()
            else:
                decision = None  # brain-only: pure neural decoder, no Jev

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
            if plasticity:
                pulse = plasticity.pending_injection()  # DAN reward pulse
                if pulse is not None:
                    for i, c in zip(*pulse):
                        combined[int(i)] = combined.get(int(i), 0.0) + float(c)
            engine.inject_input(
                np.fromiter(combined.keys(), dtype=np.int64),
                np.fromiter(combined.values(), dtype=np.float32))

            engine.step(steps_neural)
            from flydoom.motor.reflexes import aim_in_reticle
            aim = aim_in_reticle(obs, aim_cone, aim_range)
            decoded = decoder.decode(engine.rate, aim_ok=aim,
                                     spikes=getattr(engine, "counts", None))
            if weighting:
                decoded = apply_weighting(
                    decoded, weights_of(list(decoded["scores"]), decision,
                                        aim_ok=aim))
            combo = decoded.get("combo") or [decoded["selected"]]
            forced = None
            if unstuck is not None:
                forced = unstuck.update(
                    obs.position_x, obs.position_y,
                    trying_to_move=any(a in ("forward", "backward")
                                       for a in combo),
                    t_game_s=obs.episode_tic / 35.0)
                if forced:
                    combo = [forced]
                    decoded["selected"] = forced
            assisted = False
            if assist is not None and not forced:
                assisted = assist.update(bool(aim), combo)
                if assisted:
                    combo = ["attack", *combo] if combo else ["attack"]
            result = env.step(combo)
            obs = result.observation
            cleared = None
            if scenario == "fly_arena" and float(obs.kills) >= arena_enemies:
                cleared = "arena_cleared"
            elif goal is not None and math.hypot(obs.position_x - goal[0],
                                                 obs.position_y - goal[1]) <= 96.0:
                cleared = "e1m1_cleared"
            total_reward += result.reward
            health_hist.append(float(obs.health))
            if plasticity:
                ctx = {"attack": "attack" in combo, "aim_ok": aim,
                       "stuck": bool(unstuck and unstuck.stuck_now),
                       "escaped": bool(unstuck and unstuck.pop_escape())}
                plasticity.note_reward(shaped_reward(self.cfg, prev_obs, obs,
                                                     result.done, ctx=ctx,
                                                     terms=reward_terms))
                plasticity.update(engine.rate, step_period_s)
                if self._checkpoint_path \
                        and time.time() - self._last_checkpoint_t >= float(
                            self.cfg.get("plasticity", {}).get(
                                "checkpoint_interval_s", 60.0)):
                    plasticity.save_checkpoint(self._checkpoint_path,
                                               self._cfg_sha)
                    self._last_checkpoint_t = time.time()
            prev_obs = obs
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
                "profile": scenario,  # active scenario_profiles entry (per-scenario
                                      # control settings: aim cone, forward_min, unstuck)
                "controller": controller,
                "sequence": sequence,
                "generated_at_ms": round(now * 1000.0, 1),
                "frame": "data:image/jpeg;base64," + last_frame_b64,
                "game": {"health": float(obs.health), "kills": int(obs.kills),
                         "ammo": float(obs.ammo),
                         "enemies": (env.live_enemy_count()
                                     if hasattr(env, "live_enemy_count") else None),
                         "alive_s": round(obs.episode_tic / 35.0, 2),
                         "unstuck": bool(forced),
                         "episode": episode_i},
                "position": {"x": float(obs.position_x),
                             "y": float(obs.position_y)},
                "goal": goal_pub,
                "motor": {"selected": decoded["selected"],
                          "combo": combo,
                          "aim_ok": aim,
                          "attack_spiked": bool(decoded.get("attack_spiked", False)),
                          "attack_assisted": bool(assisted),
                          "turn_offset": round(float(getattr(
                              decoder, "turn_offset", 0.0)), 4),
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
                "learning": ({**plasticity.stats(),
                              "checkpoint_id": plasticity.checkpoint_id}
                             if plasticity else None),
                "jev": None if decision is None else {
                    "model": decision.model, "is_mock": decision.is_mock,
                    "probabilities": decision.probabilities,
                    "latency_ms": round(decision.latency_ms, 2),
                    "usage": decision.meta.get("usage"),
                    "intent": (decision.meta.get("choices") or {}).get("INTENT"),
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
                          "aim_ok": aim,
                          "attack_spiked": bool(decoded.get("attack_spiked", False)),
                          "attack_assisted": bool(assisted),
                          "unstuck": bool(forced),
                          "neural_scores": decoded.get("neural_scores"),
                          "jev_weights": decoded.get("jev_weights"),
                          "channels": {k: round(v, 4)
                                       for k, v in decoded.get("channels", {}).items()},
                          "imbalances": decoded.get("imbalances"),
                          "readout_rates": decoded.get("readout_rates")},
                "reward": result.reward,
                "learning": plasticity.stats() if plasticity else None,
                "controller_latency_ms": round(latencies[-1], 3),
            })
            if cleared:
                reset_reason = cleared
                break
            if result.done:
                break
            elapsed = time.perf_counter() - t_step
            if elapsed < step_period_s:
                time.sleep(step_period_s - elapsed)

        episode = {
            "controller_steps": step_i + 1,
            "controller": controller,
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
            "unstuck_triggers": unstuck.triggers if unstuck else 0,
            "unstuck_escapes": unstuck.escapes if unstuck else 0,
            "settle": settle_meta,
            "learning": ({**plasticity.stats(),
                          "delta_sha256_16": plasticity.delta_sha(),
                          "checkpoint_id": plasticity.checkpoint_id,
                          "reward_terms": reward_terms}
                         if plasticity else None),
        }
        writer.write_episode_end({"episode": episode})
        writer.close()
        if plasticity and self._checkpoint_path:
            plasticity.save_checkpoint(self._checkpoint_path, self._cfg_sha)
            self._last_checkpoint_t = time.time()
        cause = ("died (health 0)" if float(obs.health) <= 0
                 else reset_reason.replace("_", " "))
        summary = (f"survived {episode['survival_s']}s, "
                   f"{int(obs.kills)} kills, {cause}")
        scheduler.note_episode_outcome(summary)
        try:
            mp = Path("outputs/learning/jev_memory.jsonl")
            mp.parent.mkdir(parents=True, exist_ok=True)
            lines = (mp.read_text().splitlines() if mp.exists() else [])[-49:]
            lines.append(json.dumps({"run_id": run_id, "controller": controller,
                                     "scenario": scenario, "summary": summary}))
            mp.write_text("\n".join(lines) + "\n")
        except OSError:
            pass
        log.info("live episode %d done: run_id=%s tics=%d kills=%d reason=%s",
                 episode_i, run_id, int(obs.episode_tic), int(obs.kills),
                 reset_reason)
        return reset_reason

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
    def request_new_episode(self, scenario: str | None = None,
                            controller: str | None = None,
                            reset_learning: bool = False) -> str | None:
        """Abort the current episode; optionally switch scenario/controller for
        the next one. Learned weights carry over by default (one continuously-
        learning fly); reset_learning=true restarts the fly fresh.
        Returns an error string for invalid input, else None."""
        if scenario is not None:
            from flydoom.doom.base import SCENARIOS
            if scenario not in SCENARIOS:
                return (f"unknown scenario {scenario!r}; "
                        f"supported: {sorted(SCENARIOS)}")
            self._pending_scenario = scenario
        if controller is not None:
            if controller not in ("jev", "brain"):
                return (f"unknown controller {controller!r}; "
                        "supported: ['brain', 'jev']")
            self._pending_controller = controller
        self._reset_learning = bool(reset_learning)
        self._reset_requested.set()
        return None

    def health(self) -> dict:
        jev_ok = self.degraded_reason is None
        return {"status": self.status,
                "controller": self.controller,
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
        """Start a fresh episode (new seed). Optional body:
        {"scenario": name} switches the environment, {"controller":
        "jev"|"brain"} switches the controller (brain = MaleCNS alone,
        Jev bypassed, zero API calls) for the next episode.
        {"reset_learning": true} restarts the fly's learned weights fresh
        (default: they carry over — one continuously-learning fly)."""
        err = loop.request_new_episode(
            scenario=(body or {}).get("scenario"),
            controller=(body or {}).get("controller"),
            reset_learning=bool((body or {}).get("reset_learning", False)))
        if err:
            return {"status": "error", "error": err}
        return {"status": "reset_requested",
                "scenario": (body or {}).get("scenario") or "unchanged",
                "controller": (body or {}).get("controller") or "unchanged",
                "reset_learning": bool((body or {}).get("reset_learning", False))}

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
