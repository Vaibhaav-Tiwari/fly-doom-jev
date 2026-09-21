"""Environment backends: real ViZDoom and a deterministic fixture.

Both expose the same interface::

    reset() -> Observation
    step(action: str) -> StepResult
    available_actions -> list[str]
    backend_name -> str
    close()

The fixture is deterministic (seeded), clearly labeled ``fixture``, and exists
so tests and machines without a display/doom binary can still exercise the
closed loop. It is never presented as real ViZDoom validation.
"""

from .base import Observation, StepResult
from .env import FixtureEnv, VizDoomEnv, make_env

__all__ = ["Observation", "StepResult", "VizDoomEnv", "FixtureEnv", "make_env"]
