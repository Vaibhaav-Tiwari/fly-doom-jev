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
    if isinstance(connectome, FullConnectome):
        from flydoom.neural.plasticity import prepare_connectome
        connectome = prepare_connectome(cfg, connectome)  # before engine bind
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
            rate_tau_ms=float(eng.get("rate_tau_ms", 100.0)),
            drive_cap_mv=float(eng.get("drive_cap_mv", 1e9)))
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
            half_saturation=float(v.get("half_saturation", 0.02)),
            lamina_bias=float(v.get("lamina_bias", 8.0)),
            adaptation_tau_ms=float(v.get("adaptation_tau_ms", 2000.0))))
    v = cfg["vision"]
    enc = RetinaEncoder(tuple(v.get("retina_size", [16, 12])),
                        gain=float(v.get("gain", 10.0)))
    return ("mosaic", enc)


def resolve_neural_steps(cfg: dict) -> tuple[int, float, float]:
    """Neural sim steps per controller step -> (steps, dt_ms, ms_per_game_tic).

    `neural.ms_per_game_tic` (brain milliseconds simulated per game tic) is the
    primary knob: more neural time per tic lets activity propagate from the
    optic lobe into central brain and descending neurons before the next
    frame, at the cost of a slower game in wall time. Falls back to the legacy
    `steps_per_controller_step` when unset.
    """
    neural = cfg["neural"]
    dt = float(neural.get("timestep_ms", 1.0))
    frame_skip = int(cfg["environment"].get("frame_skip", 4))
    ms_per_tic = neural.get("ms_per_game_tic")
    if ms_per_tic is not None:
        steps = max(1, round(float(ms_per_tic) * frame_skip / dt))
    else:
        steps = int(neural.get("steps_per_controller_step", 32))
        ms_per_tic = steps * dt / frame_skip
    return steps, dt, float(ms_per_tic)


def run_episode(cfg: dict, record: bool = True) -> dict:
    """Run one closed-loop episode. Returns
    {"recording_dir": Path | None, "metrics": dict}."""
    import flydoom
    seed = int(cfg.get("seed", 42))

    env = make_env(cfg)
    from flydoom.config import apply_scenario_profile
    scenario = getattr(env, "scenario", None)
    cfg = apply_scenario_profile(cfg, scenario)  # per-scenario control profile
    connectome, engine = build_neural(cfg)
    from flydoom.neural.tonic import maybe_calibrate_tonic
    tonic_meta = maybe_calibrate_tonic(cfg, connectome, engine)
    vision_kind, vision = build_vision(cfg, connectome)
    jev_client = make_jev_client(cfg)
    jev_cfg = cfg["jev"]
    scheduler = JevScheduler(
        jev_client, cadence_hz=float(jev_cfg.get("cadence_hz", 3.0)),
        fast_questions=tuple(jev_cfg.get("fast_questions", ["ATTACK"])),
        strategy_period_s=jev_cfg.get("strategy_period_s"))
    controller = cfg.get("controller", "jev")
    # brain-only controller: Jev fully bypassed — scheduler stays paused,
    # zero API calls, decisions are always None (weighting is a no-op)
    scheduler.set_active(controller == "jev")
    bridge = JevBridge(connectome, cfg["bridge"]["mappings"],
                       gain=float(cfg["bridge"].get("gain", 30.0)),
                       choice_mappings=cfg["bridge"].get("choice_mappings"))
    decoder = make_decoder(connectome, cfg)
    if tonic_meta.get("enabled"):
        # decode evoked activity above the calibrated tonic baseline so
        # thresholds keep their meaning (see tonic_meta in the header)
        decoder.set_baseline(engine.rate.copy())
    from flydoom.integration.weighting import (apply_action_weighting,
                                               jev_action_weights)
    weighting = bool(cfg["jev"].get("action_weighting", False))
    intent_biases = cfg["jev"].get("intent_biases")
    from flydoom.neural.plasticity import (maybe_make_plasticity,
                                           shaped_reward)
    plasticity = maybe_make_plasticity(cfg, connectome, engine)

    neural_cfg = cfg["neural"]
    steps_neural, dt_neural, ms_per_tic = resolve_neural_steps(cfg)
    max_steps = int(cfg["environment"].get("max_controller_steps", 175))
    motor_cfg = cfg.get("motor", {})
    aim_cone = float(motor_cfg.get("attack_aim_cone_deg", 14.0))
    aim_range = float(motor_cfg.get("attack_range", 0.45))
    unstuck_cfg = cfg.get("unstuck", {})
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
            "controller": controller,  # 'jev' (Jev+MaleCNS) | 'brain' (MaleCNS only)
            "config": cfg,
            "connectome": {"provenance": connectome.provenance,
                           "n_neurons": connectome.n_neurons,
                           "n_edges": (connectome.n_edges if isinstance(connectome, FullConnectome)
                                       else connectome.weights.nnz),
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
            "jev": {"mode": cfg["jev"].get("mode", "mock"), "client": jev_client.name,
                    "cadence_hz": float(cfg["jev"].get("cadence_hz", 3.0))},
            "bridge": bridge.describe(),
            "motor": decoder.describe(),
            "plasticity": (plasticity.describe() if plasticity
                           else {"enabled": False}),
            "motor_population": {"indices": [int(i) for i in motor_idx],
                                 "body_ids": motor_ids},
            "connectome_assets": assets_ref,
            "telemetry": {"population_sample": pop_sample, "top_k": top_k},
            "warnings": _fallback_warnings(env, connectome, jev_client),
        })

    scheduler.start()
    total_reward = 0.0
    reward_terms: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    controller_latencies: list[float] = []
    beh: dict[str, list] = {"turn_imb": [], "selected": [], "visible": [],
                            "dist": [], "angle": [], "attack_ch": []}
    try:
        step_i = 0
        obs = env.reset()
        engine.reset()
        from flydoom.neural.settle import settle_episode_start
        settle_meta = settle_episode_start(
            engine, vision, vision_kind, obs.frame, decoder, connectome,
            steps_neural, dt_neural,
            float(motor_cfg.get("settle_ms", 0.0)))
        scheduler.prime(encode_state(obs, env.available_actions))
        t_start = time.time()
        realtime = bool(cfg["environment"].get("realtime", False))
        step_period_s = env.frame_skip / 35.0  # ViZDoom ticrate
        prev_obs = obs
        from collections import deque
        dmg_window = max(2, round(35.0 / env.frame_skip))  # ~1 s of game time
        health_hist: deque = deque(maxlen=dmg_window + 1)
        health_hist.append(float(obs.health))
        step_i = 0
        for step_i in range(max_steps):
            t_step = time.perf_counter()
            recent_damage = max(0.0, health_hist[0] - float(obs.health))
            state = encode_state(obs, env.available_actions,
                                 recent_damage=recent_damage)
            scheduler.update_state(state)
            decision = scheduler.get_decision()

            # visual pathway -> sensory drive
            if vision_kind == "photoreceptor":
                s_idx, s_cur = vision.sample(obs.frame, steps_neural * dt_neural)
                retina_rec = vision.retina_record(top_k)
            else:
                s_idx, s_cur = vision.sensory_drive(
                    obs.frame,
                    connectome.population_indices("visual_projection"))
                retina_rec = None
            # Jev -> bridge modulation (upstream/intermediate populations only)
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
                decoded = apply_action_weighting(
                    decoded, jev_action_weights(list(decoded["scores"]), decision,
                                                intent_biases=intent_biases,
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
            total_reward += result.reward
            health_hist.append(float(obs.health))
            if plasticity:
                ctx = {"attack": "attack" in combo, "aim_ok": aim,
                       "stuck": bool(unstuck and unstuck.stuck_now),
                       "escaped": bool(unstuck and unstuck.pop_escape())}
                plasticity.note_reward(shaped_reward(cfg, prev_obs, obs,
                                                     result.done, ctx=ctx,
                                                     terms=reward_terms))
                plasticity.update(engine.rate, step_period_s)
            prev_obs = obs
            action_counts[decoded["selected"]] = action_counts.get(decoded["selected"], 0) + 1
            if len(combo) > 1:
                action_counts["combo:" + "+".join(combo)] = \
                    action_counts.get("combo:" + "+".join(combo), 0) + 1
            latency_ms = (time.perf_counter() - t_step) * 1000.0
            controller_latencies.append(latency_ms)
            beh["turn_imb"].append(float(decoded.get("imbalances", {}).get("turn", 0.0)))
            beh["attack_ch"].append(float(decoded.get("channels", {}).get("attack", 0.0)))
            beh["selected"].append(decoded["selected"])
            beh["visible"].append(bool(state.enemy_visible))
            beh["dist"].append(float(state.enemy_distance))
            beh["angle"].append(float(state.enemy_angle))

            if writer:
                frame_ref = writer.add_frame(obs.frame)
                motor_rec = {"scores": decoded["scores"],
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
                        "choice_probabilities": decision.meta.get("choice_probabilities"),
                    },
                    "neural": {"time_ms": round(engine.time_ms, 2),
                               "steps": steps_neural,
                               "total_spikes": engine.total_spikes},
                    "retina": retina_rec,
                    "populations": engine.get_population_activity(pop_sample),
                    "activity": _activity_record(engine, motor_idx, top_k),
                    "motor": motor_rec,
                    "reward": result.reward,
                    "learning": plasticity.stats() if plasticity else None,
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
        episode = {
            "controller_steps": step_i + 1,
            "controller": controller,
            "episode_tics": int(obs.episode_tic),
            "survival_s": round(obs.episode_tic / 35.0, 2),  # 35 tics/s game time
            "duration_s": round(engine.time_ms / 1000.0, 2),
            "total_reward": round(total_reward, 3),
            "kills": int(obs.kills),
            "health_end": float(obs.health),
            "ammo_end": float(obs.ammo),
            "action_counts": action_counts,
            "jev_decisions": scheduler.decisions_made,
            "jev_errors": scheduler.errors,
            "jev_disabled_reason": scheduler.disabled_reason,
            "jev_cost_usd_total": round(scheduler.total_cost_usd, 6),
            "jev_credits_remaining_usd": scheduler.credits_remaining_usd,
            "controller_latency_ms": {
                "mean": round(float(np.mean(controller_latencies)), 3),
                "p95": round(float(np.percentile(controller_latencies, 95)), 3),
                "max": round(float(np.max(controller_latencies)), 3)}
            if controller_latencies else {},
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
        if writer:
            writer.write_episode_end({"episode": episode})
            writer.close()

    log.info("episode done: steps=%d tics=%d kills=%d reward=%.2f jev=%d decisions/%d errors",
             step_i + 1, int(obs.episode_tic), int(obs.kills), total_reward,
             scheduler.decisions_made, scheduler.errors)
    out = {"recording_dir": writer.dir if writer else None, "metrics": episode}
    if writer:
        print(f"recording: {writer.dir}")
    return out


def _behavior_metrics(beh: dict[str, list]) -> dict:
    """Honest behavior-quality metrics from per-step records."""
    n = len(beh["selected"])
    if n == 0:
        return {}
    imb = np.asarray(beh["turn_imb"])
    vis = np.asarray(beh["visible"], dtype=bool)
    dist = np.asarray(beh["dist"])
    ang = np.asarray(beh["angle"])
    sel = beh["selected"]
    seized = float(np.mean(np.abs(imb) > 0.95))
    # convention-free aim tracking: fraction of turn steps where the absolute
    # aim error |enemy_angle| actually shrank (enemy visible on both steps).
    # (ViZDoom and the fixture env have opposite angle sign conventions, so a
    # sign-based agreement metric would be misleading.)
    turns = [i for i in range(1, n)
             if sel[i] in ("turn_left", "turn_right") and vis[i] and vis[i - 1]]
    tracking = float(np.mean([abs(ang[i]) < abs(ang[i - 1]) for i in turns])) \
        if turns else None
    # attack opportunities: enemy visible and close
    opportunity = vis & (dist < 0.4)
    fired = float(np.mean([s == "attack" for s, o in zip(sel, opportunity) if o])) \
        if opportunity.any() else None
    return {
        "frac_steps_turn_saturated": round(seized, 3),
        "turn_reduces_aim_error_frac": None if tracking is None else round(tracking, 3),
        "attack_when_close_frac": None if fired is None else round(fired, 3),
        "attack_channel_mean_hz": round(float(np.mean(beh["attack_ch"])), 2),
    }


def run_episodes(cfg: dict, episodes: int) -> dict:
    """Run N recorded episodes; designate the best as the featured recording.

    Selection: survival time (game tics), then kills. All episode recordings
    are kept; recordings_root/featured.json points at the winner and lists all
    episode metrics honestly.
    """
    root = Path(cfg.get("recording", {}).get("directory", "outputs/recordings"))
    results = []
    for ep_i in range(int(episodes)):
        ep_cfg = dict(cfg, seed=int(cfg.get("seed", 42)) + ep_i)
        log.info("=== episode %d/%d (seed %d) ===", ep_i + 1, episodes,
                 ep_cfg["seed"])
        out = run_episode(ep_cfg, record=True)
        m = dict(out["metrics"])
        m["run_id"] = out["recording_dir"].name if out["recording_dir"] else None
        results.append(m)
    best = max(results, key=lambda m: (m["episode_tics"], m["kills"]))
    featured = {"format": 1,
                "selection": "max survival episode_tics, then kills",
                "featured_run_id": best["run_id"],
                "episodes": results}
    root.mkdir(parents=True, exist_ok=True)
    (root / "featured.json").write_text(json.dumps(featured, indent=2))
    log.info("featured episode: %s (tics=%d, kills=%d)",
             best["run_id"], best["episode_tics"], best["kills"])
    return {"featured": best, "episodes": results}


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
    p.add_argument("--episodes", type=int, default=None,
                   help="run N episodes, keep the best as featured "
                        "(default: recording.episodes from config, else 1)")
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    episodes = args.episodes
    if episodes is None:
        episodes = int(cfg.get("recording", {}).get("episodes", 1)) \
            if not args.no_record else 1
    if episodes > 1 and not args.no_record:
        out = run_episodes(cfg, episodes)
        print(json.dumps({"featured_recording": out["featured"]["run_id"],
                          "featured_tics": out["featured"]["episode_tics"],
                          "featured_kills": out["featured"]["kills"]}))
        return 0
    out = run_episode(cfg, record=not args.no_record)
    if out["recording_dir"]:
        print(json.dumps({"recording_dir": str(out["recording_dir"])}))
    print(json.dumps({"metrics": out["metrics"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
