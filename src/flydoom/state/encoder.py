"""Typed, versioned environment state encoder.

This is the ONLY representation Jev ever sees — raw pixels never leave the
backend. Fields are normalized to [0,1]/[-1,1] where practical so the schema is
stable across scenarios.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from flydoom.doom.base import ACTIONS, Observation

SCHEMA_VERSION = "1.0"


class EnvironmentState(BaseModel):
    schema_version: str = Field(default=SCHEMA_VERSION)
    episode_tic: int
    health: float = Field(ge=0.0)
    ammo: float = Field(ge=0.0)
    kills: int = Field(ge=0)
    enemy_visible: bool
    enemy_distance: float = Field(ge=0.0, le=1.0)   # 1.0 = far / not visible
    enemy_angle: float = Field(ge=-1.0, le=1.0)     # signed offset from heading
    threat_level: float = Field(ge=0.0, le=1.0)
    available_actions: list[str]


def compute_threat(obs: Observation) -> float:
    """Heuristic threat score in [0,1]: nearer & more centered = higher threat.

    Engineering heuristic (documented in SCIENCE.md), not a learned model.
    """
    if not obs.enemy_visible:
        return 0.0
    proximity = 1.0 - obs.enemy_distance
    centered = 1.0 - abs(obs.enemy_angle)
    return float(round(min(1.0, max(0.0, 0.7 * proximity + 0.3 * centered)), 4))


def encode_state(obs: Observation) -> EnvironmentState:
    return EnvironmentState(
        episode_tic=obs.episode_tic,
        health=float(obs.health),
        ammo=float(obs.ammo),
        kills=int(obs.kills),
        enemy_visible=bool(obs.enemy_visible),
        enemy_distance=float(obs.enemy_distance),
        enemy_angle=float(obs.enemy_angle),
        threat_level=compute_threat(obs),
        available_actions=list(ACTIONS),
    )
