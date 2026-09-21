"""Full closed-loop integration test: fixture env + fixture graph + mock Jev.

This validates the loop mechanics (env -> state/vision -> Jev -> bridge ->
engine -> decoder -> env -> recording -> replay loading). It does NOT validate
the real ViZDoom or real MaleCNS path; that is covered by the demo run.
"""

import json

from flydoom.api.replay import load_recording
from flydoom.experiments.runner import run_episode


def test_closed_loop_episode_end_to_end(fixture_cfg, tmp_path):
    out = run_episode(fixture_cfg, record=True)
    assert out is not None and (out / "recording.jsonl").exists()

    rec = load_recording(fixture_cfg["recording"]["directory"], out.name)
    h = rec["header"]
    assert h["recording_format_version"] == "2.1"
    assert h["environment"]["backend"] == "fixture"
    assert h["connectome"]["provenance"]["source"] == "fixture"
    assert any("FIXTURE" in w for w in h["warnings"])  # fallbacks are explicit

    # v2.1 header additions: fixture graph -> no shared asset bundle
    assert h["connectome_assets"] is None
    motor_pop = h["motor_population"]
    assert len(motor_pop["indices"]) == len(motor_pop["body_ids"]) > 0
    contributing = h["motor"]["contributing"]
    assert set(contributing) <= set(h["config"]["motor"]["actions"]) - {"noop"}
    for entry in contributing.values():
        assert len(entry["indices"]) == len(entry["body_ids"]) > 0
    for q, sl in h["bridge"]["slices"].items():
        assert sl["population"] == h["bridge"]["mappings"][q]
        assert sl["count"] > 0 and sl["start"] is not None

    steps = rec["steps"]
    assert len(steps) >= 10
    n_motor = len(motor_pop["indices"])
    top_k = h["telemetry"]["top_k"]
    for s in steps:
        assert "state" in s and "motor" in s and "populations" in s
        assert "frame_ref" in s and "t_ms" in s
        assert s["motor"]["selected"] in h["config"]["motor"]["actions"]
        # v2.1 per-step neuron-level activity
        act = s["activity"]
        assert len(act["top"]) <= top_k
        assert all(isinstance(i, int) and r > 0 for i, r in act["top"])
        assert len(act["motor_rates"]) == n_motor
    # Jev was primed -> decisions present from step 0
    assert steps[0]["jev"] is not None
    assert steps[0]["jev"]["is_mock"] is True
    # frames recorded as JPEG and consistent with step count
    assert rec["frames"]["count"] >= len(steps) - 1
    assert rec["frames"]["codec"] == "jpeg"
    # episode metrics present
    assert rec["summary"] is not None
    ep = rec["summary"]["episode"]
    assert ep["controller_steps"] == len(steps)
    assert ep["jev_decisions"] > 0
    # no secrets anywhere in the recording
    raw = json.dumps(rec)
    assert "api_key" not in raw.lower() and "authorization" not in raw.lower()
