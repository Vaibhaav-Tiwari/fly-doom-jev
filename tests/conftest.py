import numpy as np
import pytest

from flydoom.config import load_config


@pytest.fixture()
def fixture_cfg(tmp_path):
    """Config using ONLY deterministic fixtures (env + connectome) and tmp output."""
    cfg = load_config("configs/demo.yaml")
    cfg["environment"]["backend"] = "fixture"
    cfg["environment"]["realtime"] = False
    cfg["environment"]["max_controller_steps"] = 30
    cfg["neural"]["mode"] = "fixture"
    cfg["neural"]["steps_per_controller_step"] = 4
    cfg["jev"]["mode"] = "mock"   # tests never need an API key
    cfg["bridge"].pop("choice_mappings", None)  # type targets need the full graph
    cfg["recording"]["directory"] = str(tmp_path / "recordings")
    return cfg


@pytest.fixture()
def fixture_connectome():
    from flydoom.malecns.graph import fixture_connectome
    return fixture_connectome(seed=42)


@pytest.fixture()
def obs():
    from flydoom.doom.base import Observation
    return Observation(
        frame=np.zeros((240, 320, 3), dtype=np.uint8), health=82.0, ammo=17.0,
        kills=1, position_x=0.0, position_y=0.0, angle_deg=0.0,
        enemy_visible=True, enemy_distance=0.37, enemy_angle=0.21, episode_tic=12)
