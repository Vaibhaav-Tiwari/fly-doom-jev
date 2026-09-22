"""Environment backends."""

from __future__ import annotations

import logging
import math
import os

import numpy as np

from .base import SCENARIOS, Observation, StepResult

log = logging.getLogger(__name__)


class VizDoomEnv:
    """Real ViZDoom wrapped behind the common interface (RGB frames)."""

    backend_name = "vizdoom"

    def __init__(self, cfg: dict):
        import vizdoom as vzd

        env_cfg = cfg["environment"]
        self._label_pos: dict[int, tuple] = {}   # object_id -> last position
        self._label_still: dict[int, int] = {}   # object_id -> stationary steps
        self._visible_ids: set[int] = set()      # ids in the latest observation
        self.corpse_steps = int(env_cfg.get("corpse_steps", 40))
        self.scenario = env_cfg.get("scenario", "defend_the_center")
        if self.scenario not in SCENARIOS:
            raise ValueError(f"unknown scenario {self.scenario!r}; "
                             f"supported: {sorted(SCENARIOS)}")
        buttons, actions = SCENARIOS[self.scenario]
        g = vzd.DoomGame()
        if self.scenario == "fly_arena":
            from flydoom import scenarios as _scen_pkg
            wad_path = os.path.join(os.path.dirname(_scen_pkg.__file__),
                                    "fly_arena.wad")
            if not os.path.exists(wad_path):
                raise RuntimeError("fly_arena.wad missing; run "
                                   "`python -m flydoom.scenarios.make_fly_arena`")
            g.set_doom_map("MAP01")
        elif self.scenario == "e1m1":
            # FreeDoom phase 1 (bundled with ViZDoom as the default game WAD
            # base; BSD-3 license) — E1M1 map, the FreeDoom DOOM E1M1
            wad_path = os.path.join(os.path.dirname(vzd.__file__), "freedoom1.wad")
            if not os.path.exists(wad_path):
                raise RuntimeError("freedoom1.wad not found in the vizdoom "
                                   "package; e1m1 scenario unavailable")
            g.set_doom_game_path(wad_path)  # IWAD, not a -file scenario
            g.set_doom_map("E1M1")
        else:
            scen_dir = os.path.join(os.path.dirname(vzd.__file__), "scenarios")
            wad_path = os.path.join(scen_dir, self.scenario + ".wad")
            g.set_doom_map("MAP01")
        if self.scenario != "e1m1":
            g.set_doom_scenario_path(wad_path)
        # SELECTED_WEAPON_AMMO tracks the held weapon's pool in every scenario
        # (empirical: ViZDoom's AMMO2 read follows Clip, not shells)
        self._ammo_var = "SELECTED_WEAPON_AMMO"
        g.set_screen_resolution(getattr(
            vzd.ScreenResolution, env_cfg.get("screen_resolution", "RES_320X240")))
        g.set_screen_format(getattr(vzd.ScreenFormat,
                                    env_cfg.get("screen_format", "RGB24")))
        g.set_window_visible(False)
        g.set_labels_buffer_enabled(True)
        self._var_names = ["HEALTH", self._ammo_var, "POSITION_X", "POSITION_Y",
                           "ANGLE", "KILLCOUNT"]
        for name in self._var_names:
            g.add_available_game_variable(getattr(vzd.GameVariable, name))
        g.set_available_buttons([getattr(vzd.Button, b) for b in buttons])
        g.set_episode_timeout(int(env_cfg.get("max_episode_tics", 2100)))
        g.set_living_reward(float(env_cfg.get("living_reward", -0.01)))
        self.skill = env_cfg.get("skill")
        if self.skill is not None:
            g.set_doom_skill(int(self.skill))  # 1-5; scenario default is 3
        g.set_mode(vzd.Mode.PLAYER)
        g.set_seed(int(cfg.get("seed", 42)))  # reproducible spawns/rewards
        g.init()
        self.game = g
        self.frame_skip = int(env_cfg.get("frame_skip", 4))
        self.available_actions = list(actions)
        self._tic = 0
        self._last_obs: Observation | None = None

    def reset(self) -> Observation:
        self.game.new_episode()
        if self.scenario == "fly_arena":
            # stock defend_the_line granted infinite ammo via its ACS script;
            # our script-free WAD uses the equivalent documented cvar instead
            self.game.send_game_command("sv_infiniteammo 1")
        self._tic = 0
        self._label_pos.clear()
        self._label_still.clear()
        return self._observe()

    def step(self, action) -> StepResult:
        # action: a single action string or a list (combo: simultaneous buttons)
        actions = [action] if isinstance(action, str) else list(action)
        n_buttons = len(self.game.get_available_buttons())
        buttons = [0.0] * n_buttons
        for a in actions:
            if a != "noop":
                buttons[self.available_actions.index(a)] = 1.0
        reward = float(self.game.make_action(buttons, self.frame_skip))
        self._tic += self.frame_skip
        done = self.game.is_episode_finished()
        obs = self._observe() if not done else self._last_obs
        return StepResult(obs, reward, done, {})

    def set_seed(self, seed: int) -> None:
        """New seed for the NEXT reset (live mode: fresh spawn per episode)."""
        self.game.set_seed(int(seed))

    def live_enemy_count(self) -> int:
        """Enemies currently visible and tracked as alive (corpse-filtered)."""
        return int(sum(1 for oid in self._visible_ids
                       if self._label_still.get(oid, 0) < self.corpse_steps))

    def close(self) -> None:
        self.game.close()

    def _observe(self) -> Observation:
        g = self.game
        s = g.get_state()
        frame = s.screen_buffer
        if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
            frame = np.moveaxis(frame, 0, -1)  # (3,H,W) -> (H,W,3)
        frame = np.ascontiguousarray(frame)
        vals = {n: float(g.get_game_variable(getattr(self._vzd_var, n)))
                for n in self._var_names}
        px, py, ang = vals["POSITION_X"], vals["POSITION_Y"], vals["ANGLE"]
        enemy_visible, dist, rel_ang = False, 1.0, 0.0
        best = None
        self._visible_ids = set()
        for l in s.labels:
            if l.object_name in ("DoomPlayer",) or l.object_position_x is None:
                continue
            self._visible_ids.add(l.object_id)
            # corpses stay in the labels buffer with their living name; an
            # object that has not moved for `corpse_steps` observations is
            # treated as dead and dropped from enemy tracking. 40 steps
            # (~4.6s game time): shooting imps legitimately stand still for
            # seconds, and filtering them as corpses made Jev/blind to live
            # enemies dead-center on screen (measured bug 2026-09-22)
            pos = (round(l.object_position_x, 1), round(l.object_position_y, 1))
            if self._label_pos.get(l.object_id) == pos:
                self._label_still[l.object_id] = self._label_still.get(l.object_id, 0) + 1
            else:
                self._label_still[l.object_id] = 0
            self._label_pos[l.object_id] = pos
            if self._label_still[l.object_id] >= self.corpse_steps:
                continue
            d = math.hypot(l.object_position_x - px, l.object_position_y - py)
            if best is None or d < best[0]:
                best = (d, l)
        if best is not None:
            d, l = best
            bearing = math.degrees(math.atan2(l.object_position_y - py,
                                              l.object_position_x - px))
            rel = ((bearing - ang + 180.0) % 360.0) - 180.0
            enemy_visible = True
            dist = float(min(1.0, d / 1500.0))
            rel_ang = float(max(-1.0, min(1.0, rel / 90.0)))
        obs = Observation(frame=frame, health=vals["HEALTH"], ammo=vals[self._ammo_var],
                          kills=int(vals["KILLCOUNT"]), position_x=px,
                          position_y=py, angle_deg=ang,
                          enemy_visible=enemy_visible, enemy_distance=dist,
                          enemy_angle=rel_ang, episode_tic=self._tic)
        self._last_obs = obs
        return obs

    @property
    def _vzd_var(self):
        import vizdoom as vzd
        return vzd.GameVariable


class FixtureEnv:
    """Deterministic scripted environment with the same interface as VizDoomEnv.

    Clearly labeled ``fixture``: enemies on a 2D plane with a scripted
    approach/wobble policy and a procedurally rendered RGB frame (bright blob
    = nearest enemy). NOT ViZDoom; exists for tests and headless-less dev.
    """

    backend_name = "fixture"

    def __init__(self, cfg: dict):
        env_cfg = cfg["environment"]
        self.frame_skip = int(env_cfg.get("frame_skip", 4))
        self.max_steps = int(env_cfg.get("max_controller_steps", 175))
        self.resolution = (240, 320)  # H, W
        self.available_actions = list(SCENARIOS.get(
            env_cfg.get("scenario", "defend_the_center"),
            SCENARIOS["defend_the_center"])[1])
        self._rng = np.random.default_rng(42)
        self.reset()

    def reset(self) -> Observation:
        self._step = 0
        self.health = 100.0
        self.ammo = 50.0
        self.kills = 0
        self.angle = 0.0
        self.enemy_dist = 1.0
        self.enemy_bearing = 0.35
        self.enemy_alive = True
        self._respawn_in = 0
        return self._observe()

    def step(self, action) -> StepResult:
        actions = [action] if isinstance(action, str) else list(action)
        reward = -0.01
        for a in actions:
            reward += self._apply(a)
        return self._advance(reward)

    def _apply(self, action: str) -> float:
        reward = 0.0
        if action == "forward":
            self.enemy_dist = max(0.05, self.enemy_dist - 0.04)
        elif action == "backward":
            self.enemy_dist = min(1.0, self.enemy_dist + 0.03)
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
        return reward

    def _advance(self, reward: float) -> StepResult:
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

    def set_seed(self, seed: int) -> None:
        self._rng = np.random.default_rng(int(seed))

    def live_enemy_count(self) -> int:
        return int(self.enemy_alive)

    def _rel_bearing(self) -> float:
        return ((self.enemy_bearing - self.angle + 180.0) % 360.0) - 180.0

    def _observe(self) -> Observation:
        rel = self._rel_bearing()
        h, w = self.resolution
        frame = np.full((h, w, 3), 12, dtype=np.uint8)
        frame[h // 2:, :, 1] = 30  # dark floor gradient
        if self.enemy_alive:
            cx = int(w / 2 * (1.0 + max(-1.0, min(1.0, rel / 90.0))))
            size = int(8 + 40 * (1.0 - self.enemy_dist))
            cy = h // 2
            y0, y1 = max(0, cy - size), min(h, cy + size)
            x0, x1 = max(0, cx - size), min(w, cx + size)
            frame[y0:y1, x0:x1] = (220, 60, 40)
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
