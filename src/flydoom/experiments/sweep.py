"""Tuning sweep harness (dev tool).

Runs episodes with config overrides — mock Jev, realtime off, no recording —
and prints one metrics row per episode so gain/threshold choices are made
from measured behavior, not vibes.

    python -m flydoom.experiments.sweep \
        --override neural.engine.syn_gain=0.05 --override bridge.gain=3.0 \
        --repeat 3
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys

import yaml

from flydoom.config import load_config
from flydoom.experiments.runner import run_episode


def _set_nested(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tuning sweep over config overrides")
    p.add_argument("--config", default="configs/demo.yaml")
    p.add_argument("--override", action="append", default=[],
                   help="dotted.path=value (YAML-parsed); repeatable")
    p.add_argument("--overrides-json", default=None,
                   help='JSON object of dotted.path -> value, e.g. '
                        '\'{"bridge.choice_mappings": {...}}\'')
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="explicit seeds; overrides --repeat")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING)

    base = load_config(args.config)
    base["jev"]["mode"] = "mock"
    base["environment"]["realtime"] = False
    if args.max_steps:
        base["environment"]["max_controller_steps"] = args.max_steps
    if args.overrides_json:
        for dotted, value in json.loads(args.overrides_json).items():
            _set_nested(base, dotted, value)
    for ov in args.override:
        dotted, _, raw = ov.partition("=")
        _set_nested(base, dotted, yaml.safe_load(raw))

    seeds = args.seeds or [int(base.get("seed", 42)) + i for i in range(args.repeat)]
    label = " ".join(args.override) or (args.overrides_json or "(base)")
    print(f"# overrides: {label}")
    print(f"{'seed':>5} {'tics':>5} {'surv_s':>6} {'kills':>5} {'hp':>4} {'ammo':>4} "
          f"{'seiz%':>6} {'trk%':>5} {'atk%':>5} {'atk_hz':>7}  actions")
    for seed in seeds:
        cfg = copy.deepcopy(base)
        cfg["seed"] = seed
        out = run_episode(cfg, record=False)
        m = out["metrics"]
        b = m.get("behavior", {})
        def pct(x):
            return f"{100*x:5.0f}" if x is not None else "    -"
        print(f"{seed:>5} {m['episode_tics']:>5} {m['survival_s']:>6.1f} "
              f"{m['kills']:>5} {m['health_end']:>4.0f} {m['ammo_end']:>4.0f} "
              f"{pct(b.get('frac_steps_turn_saturated')):>6} "
              f"{pct(b.get('turn_reduces_aim_error_frac')):>5} "
              f"{pct(b.get('attack_when_close_frac')):>5} "
              f"{b.get('attack_channel_mean_hz', 0.0):>7.2f}  {m['action_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
