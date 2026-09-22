"""Typed, versioned environment state encoder.

This is the ONLY representation Jev ever sees — raw pixels never leave the
backend. Fields are normalized to [0,1]/[-1,1] where practical so the schema is
stable across scenarios.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from flydoom.doom.base import Observation

SCHEMA_VERSION = "1.2"
DEFAULT_ACTIONS = ["turn_left", "turn_right", "attack", "noop"]


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
    # 1.2 additions — the Jev-facing strategy view (additive; the legacy
    # normalized fields above stay for the recording contract):
    enemy_in_view: bool = False         # alias of enemy_visible (Jev wording)
    aim_offset_deg: float = 0.0         # enemy_angle in degrees (+ = off-center)
    alive_s: float = 0.0                # episode_tic / 35 (game seconds)
    recent_damage_taken: float = 0.0    # hp lost over the last ~1 s of game time
    position_x: float = 0.0             # map units (movement/parking evidence)
    position_y: float = 0.0


def compute_threat(obs: Observation) -> float:
    """Heuristic threat score in [0,1]: nearer & more centered = higher threat.

    Engineering heuristic (documented in SCIENCE.md), not a learned model.
    """
    if not obs.enemy_visible:
        return 0.0
    proximity = 1.0 - obs.enemy_distance
    centered = 1.0 - abs(obs.enemy_angle)
    return float(round(min(1.0, max(0.0, 0.7 * proximity + 0.3 * centered)), 4))


def encode_state(obs: Observation, available_actions: list[str] | None = None,
                 recent_damage: float = 0.0) -> EnvironmentState:
    return EnvironmentState(
        episode_tic=obs.episode_tic,
        health=float(obs.health),
        ammo=float(obs.ammo),
        kills=int(obs.kills),
        enemy_visible=bool(obs.enemy_visible),
        enemy_distance=float(obs.enemy_distance),
        enemy_angle=float(obs.enemy_angle),
        threat_level=compute_threat(obs),
        available_actions=list(available_actions or DEFAULT_ACTIONS),
        enemy_in_view=bool(obs.enemy_visible),
        aim_offset_deg=round(float(obs.enemy_angle) * 90.0, 1),
        alive_s=round(obs.episode_tic / 35.0, 2),
        recent_damage_taken=round(max(0.0, float(recent_damage)), 1),
        position_x=round(float(obs.position_x), 1),
        position_y=round(float(obs.position_y), 1),
    )
