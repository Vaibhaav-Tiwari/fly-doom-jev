"""Batch experiment runner: seeds x scenarios x controllers, pure Python.

    python -m flydoom.experiments.batch --config configs/demo.yaml \
        --seeds 42 43 --scenarios fly_arena e1m1 --controllers jev brain

Zero LLM involvement at runtime; Jev API calls happen only in 'jev'
controller arms (intent cadence keeps them cheap). One brain instance
(connectome + engine + tonic calibration) is built ONCE per arm process and
reused across the arm's episodes (engine.reset() between episodes).

Output under outputs/batch/<timestamp>/:
  episodes.jsonl  one result per episode (seed, scenario, controller,
                  survival_s, kills, damage_taken_hp, attack_count,
                  unstuck_triggers, jev stats, learning stats when --learn)
  summary.csv / summary.json  means grouped by scenario x controller

Plasticity is OFF by default in batch runs (clean controller comparison);
--learn enables it per arm (learned weights persist across the arm's
episodes, fresh between arms — same policy as the live server).
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from flydoom.config import apply_scenario_profile, load_config
from flydoom.doom import make_env
from flydoom.experiments.runner import (build_neural, build_vision,
                                        resolve_neural_steps)
from flydoom.integration import JevBridge
from flydoom.integration.weighting import (apply_action_weighting,
                                           jev_action_weights)
from flydoom.jev import JevScheduler, make_jev_client
from flydoom.motor import make_decoder
from flydoom.motor.reflexes import UnstuckReflex, aim_in_reticle
from flydoom.state import encode_state

log = logging.getLogger(__name__)


def run_arm_episode(cfg: dict, scen_cfg: dict, env, engine, vision_kind,
                    vision, jev_client, bridge, decoder, plasticity,
                    steps_neural: int, dt_neural: float, seed: int) -> dict:
    """One unrecorded episode on the shared brain. Returns the result row."""
    from flydoom.neural.plasticity import shaped_reward
    controller = cfg.get("controller", "jev")
    jev_cfg = cfg["jev"]
    scheduler = JevScheduler(
        jev_client, cadence_hz=float(jev_cfg.get("cadence_hz", 3.0)),
        fast_questions=tuple(jev_cfg.get("fast_questions", ["ATTACK"])),
        strategy_period_s=jev_cfg.get("strategy_period_s"))
    scheduler.set_active(controller == "jev")  # brain arm: zero API calls
    weighting = bool(jev_cfg.get("action_weighting", False))
    intent_biases = jev_cfg.get("intent_biases")
    motor_cfg = scen_cfg.get("motor", {})
    aim_cone = float(motor_cfg.get("attack_aim_cone_deg", 14.0))
    aim_range = float(motor_cfg.get("attack_range", 0.45))
    unstuck = None
    unstuck_cfg = scen_cfg.get("unstuck", {})
    if unstuck_cfg.get("enabled"):
        unstuck = UnstuckReflex(window_s=float(unstuck_cfg.get("window_s", 2.5)),
                                epsilon=float(unstuck_cfg.get("epsilon", 6.0)),
                                turn_s=float(unstuck_cfg.get("turn_s", 1.0)))
    max_steps = int(cfg["environment"].get("max_controller_steps", 900))
    step_period_s = env.frame_skip / 35.0

    scheduler.start()
    t0 = time.time()
    attack_count = attack_aimed = 0
    damage_taken = 0.0
    total_reward = 0.0
    reward_terms: dict[str, int] = {}
    try:
        if hasattr(env, "set_seed"):
            env.set_seed(seed)
        obs = env.reset()
        engine.reset()
        if unstuck:
            unstuck.reset()
        scheduler.prime(encode_state(obs, env.available_actions))
        dmg_window = max(2, round(35.0 / env.frame_skip))
        health_hist: deque = deque(maxlen=dmg_window + 1)
        health_hist.append(float(obs.health))
        prev_obs = obs
        step_i = 0
        for step_i in range(max_steps):
            recent_damage = max(0.0, health_hist[0] - float(obs.health))
            state = encode_state(obs, env.available_actions,
                                 recent_damage=recent_damage)
            scheduler.update_state(state)
            decision = scheduler.get_decision()

            if vision_kind == "photoreceptor":
                s_idx, s_cur = vision.sample(obs.frame, steps_neural * dt_neural)
            else:
                s_idx, s_cur = vision.sensory_drive(
                    obs.frame,
                    engine.connectome.population_indices("visual_projection"))
            b_idx, b_cur = bridge.modulation_currents(decision)
            combined: dict[int, float] = {}
            for i, c in zip(s_idx, s_cur):
                combined[int(i)] = combined.get(int(i), 0.0) + float(c)
            for i, c in zip(b_idx, b_cur):
                combined[int(i)] = combined.get(int(i), 0.0) + float(c)
            if plasticity:
                pulse = plasticity.pending_injection()
                if pulse is not None:
                    for i, c in zip(*pulse):
                        combined[int(i)] = combined.get(int(i), 0.0) + float(c)
            engine.inject_input(
                np.fromiter(combined.keys(), dtype=np.int64),
                np.fromiter(combined.values(), dtype=np.float32))
            engine.step(steps_neural)

            aim = aim_in_reticle(obs, aim_cone, aim_range)
            decoded = decoder.decode(engine.rate, aim_ok=aim)
            if weighting:
                decoded = apply_action_weighting(
                    decoded, jev_action_weights(list(decoded["scores"]), decision,
                                                intent_biases=intent_biases))
            combo = decoded.get("combo") or [decoded["selected"]]
            if unstuck is not None:
                forced = unstuck.update(
                    obs.position_x, obs.position_y,
                    trying_to_move=any(a in ("forward", "backward") for a in combo),
                    t_game_s=obs.episode_tic / 35.0)
                if forced:
                    combo = [forced]
                    decoded["selected"] = forced
            result = env.step(combo)
            prev_health = float(obs.health)
            obs = result.observation
            total_reward += result.reward
            damage_taken += max(0.0, prev_health - float(obs.health))
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
            if "attack" in combo:
                attack_count += 1
                if aim:
                    attack_aimed += 1
            if result.done:
                break
    finally:
        scheduler.stop()
    wall_s = time.time() - t0
    row = {
        "seed": seed,
        "scenario": scen_cfg["environment"].get("scenario", "fly_arena"),
        "controller": controller,
        "survival_s": round(obs.episode_tic / 35.0, 2),
        "kills": int(obs.kills),
        "damage_taken_hp": round(damage_taken, 1),  # damage DEALT is not a
        # game variable in these scenarios; kills are its proxy (documented)
        "health_end": round(float(obs.health), 1),
        "total_reward": round(total_reward, 3),
        "attack_count": attack_count,
        "attack_aimed_count": attack_aimed,
        "unstuck_triggers": unstuck.triggers if unstuck else 0,
        "unstuck_escapes": unstuck.escapes if unstuck else 0,
        "controller_steps": step_i + 1,
        "wall_s": round(wall_s, 1),
        "jev_decisions": scheduler.decisions_made,
        "jev_errors": scheduler.errors,
        "jev_cost_usd": round(scheduler.total_cost_usd, 6),
    }
    if plasticity:
        row["learning"] = {**plasticity.stats(),
                           "delta_sha256_16": plasticity.delta_sha(),
                           "reward_terms": reward_terms}
    return row


def run_arm(cfg_path: str, scenario: str, controller: str, seeds: list[int],
            learn: bool, max_steps: int | None) -> list[dict]:
    """One arm (scenario x controller) over all seeds; brain built once."""
    cfg = load_config(cfg_path)
    cfg["environment"]["scenario"] = scenario
    cfg["environment"]["realtime"] = False  # batch runs as fast as the sim allows
    cfg["controller"] = controller
    if max_steps:
        cfg["environment"]["max_controller_steps"] = int(max_steps)
    if not learn:
        cfg.setdefault("plasticity", {})["enabled"] = False
    scen_cfg = apply_scenario_profile(cfg, scenario)

    env = make_env(cfg)
    try:
        connectome, engine = build_neural(cfg)
        from flydoom.neural.tonic import maybe_calibrate_tonic
        tonic_meta = maybe_calibrate_tonic(cfg, connectome, engine)
        vision_kind, vision = build_vision(cfg, connectome)
        jev_client = make_jev_client(cfg)
        bridge = JevBridge(connectome, cfg["bridge"]["mappings"],
                           gain=float(cfg["bridge"].get("gain", 30.0)),
                           choice_mappings=cfg["bridge"].get("choice_mappings"))
        decoder = make_decoder(connectome, scen_cfg)
        if tonic_meta.get("enabled"):
            decoder.set_baseline(engine.rate.copy())
        plasticity = None
        if learn:
            from flydoom.neural.plasticity import maybe_make_plasticity
            plasticity = maybe_make_plasticity(cfg, connectome, engine)
        steps_neural, dt_neural, _ = resolve_neural_steps(cfg)
        rows = []
        for seed in seeds:
            log.info("=== arm %s/%s seed %d ===", scenario, controller, seed)
            row = run_arm_episode(cfg, scen_cfg, env, engine, vision_kind,
                                  vision, jev_client, bridge, decoder,
                                  plasticity, steps_neural, dt_neural, seed)
            log.info("episode: survival=%.1fs kills=%d attacks=%d wall=%.0fs",
                     row["survival_s"], row["kills"], row["attack_count"],
                     row["wall_s"])
            rows.append(row)
        return rows
    finally:
        env.close()


def summarize(rows: list[dict]) -> list[dict]:
    """Means grouped by scenario x controller."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["scenario"], r["controller"]), []).append(r)
    out = []
    for (scenario, controller), rs in sorted(groups.items()):
        n = len(rs)
        def mean(k):
            return round(float(np.mean([r[k] for r in rs])), 2)
        out.append({
            "scenario": scenario, "controller": controller, "episodes": n,
            "survival_s_mean": mean("survival_s"),
            "survival_s_max": max(r["survival_s"] for r in rs),
            "kills_mean": mean("kills"),
            "kills_total": sum(r["kills"] for r in rs),
            "attack_count_mean": mean("attack_count"),
            "attack_aimed_frac": round(
                sum(r["attack_aimed_count"] for r in rs)
                / max(1, sum(r["attack_count"] for r in rs)), 3),
            "unstuck_triggers_mean": mean("unstuck_triggers"),
            "damage_taken_hp_mean": mean("damage_taken_hp"),
            "jev_decisions_mean": mean("jev_decisions"),
            "jev_errors_total": sum(r["jev_errors"] for r in rs),
            "jev_cost_usd_total": round(sum(r["jev_cost_usd"] for r in rs), 6),
        })
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Batch runner: seeds x scenarios "
                                            "x controllers")
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43])
    p.add_argument("--scenarios", nargs="+", default=["fly_arena", "e1m1"])
    p.add_argument("--controllers", nargs="+", default=["jev", "brain"])
    p.add_argument("--learn", action="store_true",
                   help="enable dopamine plasticity per arm (weights persist "
                        "across the arm's episodes, fresh between arms)")
    p.add_argument("--steps", type=int, default=None,
                   help="override environment.max_controller_steps")
    p.add_argument("--parallel", type=int, default=1,
                   help="arms run in parallel worker processes (each builds "
                        "its own brain; watch RAM)")
    p.add_argument("--out", default=None, help="output directory "
                   "(default outputs/batch/<timestamp>)")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    out_dir = Path(args.out) if args.out else Path(
        "outputs/batch") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [(s, c) for s in args.scenarios for c in args.controllers]
    log.info("batch: %d arms x %d seeds -> %s", len(arms), len(args.seeds),
             out_dir)
    rows: list[dict] = []
    if args.parallel > 1 and len(arms) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.parallel) as ex:
            futs = [ex.submit(run_arm, args.config, s, c, args.seeds,
                              args.learn, args.steps) for s, c in arms]
            for f in futs:
                rows.extend(f.result())
    else:
        for s, c in arms:
            rows.extend(run_arm(args.config, s, c, args.seeds, args.learn,
                                args.steps))

    with (out_dir / "episodes.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    summary = summarize(rows)
    (out_dir / "summary.json").write_text(json.dumps(
        {"config": args.config, "seeds": args.seeds, "learn": args.learn,
         "arms": [f"{s}/{c}" for s, c in arms], "groups": summary}, indent=2))
    with (out_dir / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(json.dumps({"out_dir": str(out_dir), "episodes": len(rows)}))
    for g in summary:
        print(f"{g['scenario']:>10} {g['controller']:>6}: "
              f"survival {g['survival_s_mean']:.1f}s "
              f"(max {g['survival_s_max']:.1f}), kills {g['kills_mean']:.1f}, "
              f"attacks {g['attack_count_mean']:.0f} "
              f"(aimed {g['attack_aimed_frac']:.0%}), "
              f"unstuck {g['unstuck_triggers_mean']:.1f}, "
              f"jev {g['jev_decisions_mean']:.0f} dec/{g['jev_errors_total']} err")
    return 0


if __name__ == "__main__":
    sys.exit(main())
