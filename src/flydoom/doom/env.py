"""Environment backends."""

from __future__ import annotations

import logging
import math
import os

import numpy as np

from .base import ACTIONS, Observation, StepResult

log = logging.getLogger(__name__)


class VizDoomEnv:
    """Real ViZDoom 'basic' scenario wrapped behind the common interface."""

    backend_name = "vizdoom"

    def __init__(self, cfg: dict):
        import vizdoom as vzd

        env_cfg = cfg["environment"]
        self._vzd = vzd
        g = vzd.DoomGame()
        scen_dir = os.path.join(os.path.dirname(vzd.__file__), "scenarios")
        g.set_doom_scenario_path(os.path.join(scen_dir, env_cfg.get("scenario", "basic") + ".wad"))
        g.set_doom_map("MAP01")
        g.set_screen_resolution(getattr(vzd.ScreenResolution, env_cfg.get("screen_resolution", "RES_160X120")))
        g.set_screen_format(getattr(vzd.ScreenFormat, env_cfg.get("screen_format", "GRAY8")))
        g.set_window_visible(False)
        g.set_labels_buffer_enabled(True)
        for var in (vzd.GameVariable.HEALTH, vzd.GameVariable.AMMO2,
                    vzd.GameVariable.POSITION_X, vzd.GameVariable.POSITION_Y,
                    vzd.GameVariable.ANGLE, vzd.GameVariable.KILLCOUNT):
            g.add_available_game_variable(var)
        g.set_available_buttons([vzd.Button.MOVE_FORWARD, vzd.Button.TURN_LEFT,
                                 vzd.Button.TURN_RIGHT, vzd.Button.ATTACK])
        g.set_episode_timeout(env_cfg.get("max_episode_tics", 2100))
        g.set_living_reward(-0.01)
        g.set_mode(vzd.Mode.PLAYER)
        g.init()
        self.game = g
        self.frame_skip = int(env_cfg.get("frame_skip", 4))
        self.available_actions = list(ACTIONS)
        self._tic = 0

    def reset(self) -> Observation:
        self.game.new_episode()
        self._tic = 0
        return self._observe()

    def step(self, action: str) -> StepResult:
        buttons = [0.0] * 4
        if action == "forward":
            buttons[0] = 1.0
        elif action == "turn_left":
            buttons[1] = 1.0
        elif action == "turn_right":
            buttons[2] = 1.0
        elif action == "attack":
            buttons[3] = 1.0
        reward = float(self.game.make_action(buttons, self.frame_skip))
        self._tic += self.frame_skip
        done = self.game.is_episode_finished()
        obs = self._observe() if not done else self._last_obs
        return StepResult(obs, reward, done, {})

    def close(self) -> None:
        self.game.close()

    def _observe(self) -> Observation:
        g = self.game
        s = g.get_state()
        frame = s.screen_buffer.astype(np.float32) / 255.0
        health, ammo, px, py, ang, kills = [float(v) for v in s.game_variables]
        enemy_visible, dist, rel_ang = False, 1.0, 0.0
        best = None
        for l in s.labels:
            if l.object_name not in ("DoomPlayer",) and l.object_position_x is not None:
                d = math.hypot(l.object_position_x - px, l.object_position_y - py)
                if best is None or d < best[0]:
                    best = (d, l)
        if best is not None:
            d, l = best
            bearing = math.degrees(math.atan2(l.object_position_y - py, l.object_position_x - px))
            rel = ((bearing - ang + 180.0) % 360.0) - 180.0
            enemy_visible = True
            dist = float(min(1.0, d / 1500.0))       # normalize by scenario scale
            rel_ang = float(max(-1.0, min(1.0, rel / 90.0)))
        obs = Observation(frame=frame, health=health, ammo=ammo, kills=int(kills),
                          position_x=px, position_y=py, angle_deg=ang,
                          enemy_visible=enemy_visible, enemy_distance=dist,
                          enemy_angle=rel_ang, episode_tic=self._tic)
        self._last_obs = obs
        return obs


class FixtureEnv:
    """Deterministic scripted environment with the same interface as VizDoomEnv.

    Clearly labeled ``fixture``: a single enemy on a 2D plane, a scripted
    approach/reposition policy, and a procedurally rendered grayscale frame
    (bright blob = enemy). NOT ViZDoom; exists for tests and headless-less dev.
    """

    backend_name = "fixture"

    def __init__(self, cfg: dict):
        env_cfg = cfg["environment"]
        self.frame_skip = int(env_cfg.get("frame_skip", 4))
        self.max_steps = int(env_cfg.get("max_controller_steps", 175))
        self.resolution = (120, 160)  # H, W
        self.available_actions = list(ACTIONS)
        self._rng = np.random.default_rng(42)
        self.reset()

    def reset(self) -> Observation:
        self._step = 0
        self.health = 100.0
        self.ammo = 50.0
        self.kills = 0
        self.angle = 0.0              # player heading, degrees
        self.enemy_dist = 1.0         # normalized
        self.enemy_bearing = 0.35     # degrees offset from heading
        self.enemy_alive = True
        self._respawn_in = 0
        return self._observe()

    def step(self, action: str) -> StepResult:
        reward = -0.01
        if action == "forward":
            self.enemy_dist = max(0.05, self.enemy_dist - 0.04)
        elif action == "turn_left":
            self.angle -= 15.0
        elif action == "turn_right":
            self.angle += 15.0
        elif action == "attack" and self.ammo > 0:
            self.ammo -= 1
            rel = abs(self._rel_bearing())
            if self.enemy_alive and rel < 12.0 and self.enemy_dist < 0.6:
                self.enemy_alive = False
                self.kills += 1
                reward = 1.0
                self._respawn_in = 25
        # scripted enemy behaviour: drifts toward the player, wobbles sideways
        if self.enemy_alive:
            self.enemy_dist = min(1.0, self.enemy_dist + 0.006)
            self.enemy_bearing += float(self._rng.standard_normal() * 3.0)
            if self.enemy_dist < 0.25:
                self.health = max(0.0, self.health - 0.8)
        else:
            self._respawn_in -= 1
            if self._respawn_in <= 0:
                self.enemy_alive = True
                self.enemy_dist = 1.0
                self.enemy_bearing = float(self._rng.uniform(-50, 50))
        self._step += 1
        done = self._step >= self.max_steps or self.health <= 0
        return StepResult(self._observe(), reward, done, {})

    def close(self) -> None:
        pass

    def _rel_bearing(self) -> float:
        return ((self.enemy_bearing - self.angle + 180.0) % 360.0) - 180.0

    def _observe(self) -> Observation:
        rel = self._rel_bearing()
        frame = np.full(self.resolution, 0.05, dtype=np.float32)
        if self.enemy_alive:
            # draw enemy as a bright blob: screen x from bearing, size from distance
            cx = int(self.resolution[1] / 2 * (1.0 + max(-1.0, min(1.0, rel / 90.0))))
            size = int(6 + 26 * (1.0 - self.enemy_dist))
            cy = self.resolution[0] // 2
            y0, y1 = max(0, cy - size), min(self.resolution[0], cy + size)
            x0, x1 = max(0, cx - size), min(self.resolution[1], cx + size)
            frame[y0:y1, x0:x1] = 0.9
        return Observation(
            frame=frame, health=self.health, ammo=self.ammo, kills=self.kills,
            position_x=0.0, position_y=0.0, angle_deg=self.angle,
            enemy_visible=self.enemy_alive,
            enemy_distance=self.enemy_dist if self.enemy_alive else 1.0,
            enemy_angle=max(-1.0, min(1.0, rel / 90.0)) if self.enemy_alive else 0.0,
            episode_tic=self._step * self.frame_skip,
        )


def make_env(cfg: dict):
    """Build the environment per config, with an explicit labeled fallback."""
    backend = cfg["environment"].get("backend", "vizdoom")
    if backend == "fixture":
        log.warning("environment backend=fixture: deterministic test env, NOT ViZDoom")
        return FixtureEnv(cfg)
    try:
        return VizDoomEnv(cfg)
    except Exception as exc:  # vizdoom missing / cannot init on this machine
        if backend == "vizdoom":
            log.warning("ViZDoom unavailable (%s); falling back to the deterministic "
                        "fixture environment. The real closed loop was NOT validated "
                        "in this run.", exc)
            return FixtureEnv(cfg)
        raise
