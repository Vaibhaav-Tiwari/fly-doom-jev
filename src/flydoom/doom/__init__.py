"""Environment backends: real ViZDoom and a deterministic fixture.

Both expose the same interface::

    reset() -> Observation
    step(action: str) -> StepResult
    available_actions -> list[str]
    backend_name -> str
    close()

Frames are RGB uint8 (H, W, 3) in both backends.

The fixture is deterministic (seeded), clearly labeled ``fixture``, and exists
so tests can exercise the loop without a DOOM binary. It is never presented as
real ViZDoom validation.
"""

from .base import Observation, StepResult
from .env import FixtureEnv, VizDoomEnv, make_env

__all__ = ["Observation", "StepResult", "VizDoomEnv", "FixtureEnv", "make_env"]
