"""PPO bot controller — integrates the PPO actor-critic policy with the game world.

Replaces BotController for PPO training.  All bots share one policy and one
rollout buffer.  The controller:
  1. Builds batched observations using the same v2 perception as NEAT bots.
  2. Samples actions from the PPO policy (ActorCritic).
  3. Computes per-step scalar rewards from mass-change signals.
  4. Stores (obs, action, logprob, reward, done, value) into the RolloutBuffer.
  5. Exposes collect_rollout_step() to be called once per tick from the
     training world, and finish_rollout() to trigger the PPO update.
"""
from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from . import config
from .bot import (
    _build_inputs_batch_v2,
    _V2_HALF_DIAG,
    _TARGET_DIST,
    _MAX_CELLS,
    _VEL_NORM,
    _V2_MAX_MASS_LOG,
    _V2_MERGE_TIMER_MAX,
)
from .ppo_agent import ActorCritic, RolloutBuffer, PPOTrainer, N_OBS

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Observation sanity-check specification
# Each entry: (slice_or_index, lo, hi, name)
# ---------------------------------------------------------------------------
_OBS_CHECKS: list[tuple] = [
    (slice(0, 5),   0.0, 1.0,  "interoceptive[0:5]"),
    # 8 rays × 7 channels; all should be [0,1] except prey_mass [0,~0.8]
    (slice(5, 61),  0.0, 1.0,  "rays[5:61]"),
    (slice(61, 65), 0.0, 1.0,  "quad_food[61:65]"),
    (slice(65, 69), 0.0, 1.0,  "quad_threat[65:69]"),
    (slice(69, 72), 0.0, 1.0,  "contextual[69:72]"),
    (slice(72, 74), -1.0, 1.0, "velocity[72:74]"),
    # 16 cells × 3: (rel_x, rel_y) ∈ [-1,1], mass ∈ [0,1] — all within [-1,1]
    (slice(74, 122), -1.0, 1.0, "cell_body[74:122]"),
]


def check_obs(obs: np.ndarray, label: str = "") -> bool:
    """Validate a (B, N_OBS) observation array.

    Logs a WARNING for each channel group that has out-of-range or non-finite
    values and returns True if everything is clean, False otherwise.
    """
    prefix = f"[obs check{' ' + label if label else ''}]"
    ok = True

    nan_mask = ~np.isfinite(obs)
    if nan_mask.any():
        locs = np.argwhere(nan_mask)
        log.warning("%s NaN/Inf in %d cells — first few: %s", prefix, len(locs), locs[:5].tolist())
        ok = False

    for sel, lo, hi, name in _OBS_CHECKS:
        chunk = obs[:, sel]
        c_min = float(chunk.min())
        c_max = float(chunk.max())
        if c_min < lo - 1e-4 or c_max > hi + 1e-4:
            log.warning(
                "%s %s out of [%.2f, %.2f]: min=%.4f max=%.4f mean=%.4f",
                prefix, name, lo, hi, c_min, c_max, float(chunk.mean()),
            )
            ok = False

    if ok:
        log.debug("%s all %d channels OK (B=%d)", prefix, obs.shape[1], obs.shape[0])
    return ok

if TYPE_CHECKING:
    from .game import GameWorld


class PPOBotController:
    """Drives a fixed population of bots using a shared PPO actor-critic policy.

    Usage (from training world):
        ctrl = PPOBotController(n_bots, model, trainer, buffer)
        # each tick:
        ctrl.update(world, dt)
        # when rollout is full (ctrl.step >= ctrl.n_steps):
        ctrl.finish_rollout(world)
        ctrl.reset_rollout()
    """

    def __init__(
        self,
        n_bots:  int,
        model:   ActorCritic,
        trainer: PPOTrainer | None = None,
        buffer:  RolloutBuffer | None = None,
        device:  torch.device | str = "cpu",
        inference_only: bool = False,
        obs_check_interval: int = 200,
    ) -> None:
        self.n_bots  = n_bots
        self.model   = model
        self.trainer = trainer
        self.buffer  = buffer
        self.device  = torch.device(device)
        self.inference_only = inference_only
        self.step    = 0          # current position within the rollout
        self.n_steps = buffer.n_steps if buffer is not None else 0
        self.obs_check_interval = obs_check_interval  # run check_obs every N steps; 0 = disabled
        self._obs_check_counter = 0

        # player_id -> per-bot state dict (same keys as NEAT BotController)
        self._state: dict[int, dict] = {}
        # ordered list of active bot player_ids (index matches rollout buffer)
        self._pid_order: list[int] = []
        # last observations (B, N_OBS) — needed for bootstrapping value
        self._last_obs: np.ndarray | None = None
        # last done flags (B,) — needed for bootstrapping
        self._last_dones: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Bot lifecycle
    # ------------------------------------------------------------------

    def register(self, player_id: int, start_x: float, start_y: float) -> None:
        self._state[player_id] = {
            'start_mass':             20.0,
            'ticks_alive':            0,
            'peak_mass':              20.0,
            'total_mass_accumulated': 20.0,
            'last_split_tick':        0,
            'last_pos':               (start_x, start_y),
            'distance_traveled':      0.0,
            'last_mass':              20.0,
            'food_eaten_count':       0,
            'cells_eaten_mass':       0.0,
            'idle_ticks':             0,
            'edge_ticks':             0,
            'corner_ticks':           0,
            'last_vel':               (0.0, 0.0),
            'game_time_alive':        0.0,
            'died_this_tick':         False,
        }
        if player_id not in self._pid_order:
            self._pid_order.append(player_id)

    def unregister(self, player_id: int) -> float:
        """Remove bot; return its final fitness."""
        bs = self._state.pop(player_id, None)
        if player_id in self._pid_order:
            self._pid_order.remove(player_id)
        if bs is None:
            return 0.0
        return self._compute_fitness(bs)

    def clear(self) -> None:
        """Remove all bots (called at rollout reset or world teardown)."""
        self._state.clear()
        self._pid_order.clear()

    # ------------------------------------------------------------------
    # Main tick update
    # ------------------------------------------------------------------

    def update(self, world: 'GameWorld', dt: float) -> None:
        """Called once per game tick.  Builds obs, samples action, applies to
        game, computes reward, and stores one step into the rollout buffer."""
        if not self._state:
            return
        if not self.inference_only and self.step >= self.n_steps:
            return

        dt_scale = dt / config.TICK_INTERVAL

        # ---- Active bot snapshot ----
        plan_pids = [pid for pid in self._pid_order if world.players.get(pid) is not None]
        if not plan_pids:
            return

        B = len(plan_pids)
        cw = config.WORLD_W
        ch = config.WORLD_H
        cw_h = cw * 0.5
        ch_h = ch * 0.5
        min_split = config.MIN_SPLIT_MASS
        min_eject = config.EJECT_MIN_MASS
        merge_max = _V2_MERGE_TIMER_MAX
        max_mass_log = _V2_MAX_MASS_LOG

        bot_alive   = np.zeros(B, dtype=bool)
        bot_cx      = np.zeros(B, dtype=np.float64)
        bot_cy      = np.zeros(B, dtype=np.float64)
        bot_mass    = np.zeros(B, dtype=np.float64)
        bot_largest = np.zeros(B, dtype=np.float64)
        bot_scan    = np.zeros(B, dtype=np.float64)
        intero      = np.zeros((B, 5), dtype=np.float64)
        last_vx     = np.zeros(B, dtype=np.float64)
        last_vy     = np.zeros(B, dtype=np.float64)
        rewards     = np.zeros(B, dtype=np.float32)
        dones       = np.zeros(B, dtype=np.float32)

        for i, pid in enumerate(plan_pids):
            bs     = self._state.get(pid)
            player = world.players.get(pid)
            if bs is None or player is None:
                continue
            cells = player.cells
            if not cells:
                continue

            total_mass = 0.0
            wx = 0.0
            wy = 0.0
            largest = 0.0
            max_merge = 0.0
            sr = 0.0
            er = 0.0
            for c in cells:
                m = c.mass
                total_mass += m
                wx += c.x * m
                wy += c.y * m
                if m > largest:
                    largest = m
                if c.merge_timer > max_merge:
                    max_merge = c.merge_timer
                if m >= min_split:
                    sr = 1.0
                if m >= min_eject:
                    er = 1.0
            n_cells = len(cells)
            cx = wx / total_mass if total_mass > 0.0 else cw_h
            cy_val = wy / total_mass if total_mass > 0.0 else ch_h

            bs['ticks_alive'] += 1
            bs['game_time_alive'] = bs.get('game_time_alive', 0.0) + dt
            if total_mass > bs['peak_mass']:
                bs['peak_mass'] = total_mass
            bs['total_mass_accumulated'] += total_mass

            last_x, last_y = bs['last_pos']
            vx = cx - last_x
            vy = cy_val - last_y
            bs['last_vel'] = (vx, vy)
            dist_moved = math.hypot(vx, vy)
            bs['distance_traveled'] += dist_moved
            bs['last_pos'] = (cx, cy_val)

            if dist_moved < 1.5:
                bs['idle_ticks'] += 1
            if cx < 500.0 or cx > cw - 500.0 or cy_val < 500.0 or cy_val > ch - 500.0:
                bs['edge_ticks'] += 1
            if (math.hypot(cx, cy_val) < 2000.0 or math.hypot(cw - cx, cy_val) < 2000.0
                    or math.hypot(cx, ch - cy_val) < 2000.0
                    or math.hypot(cw - cx, ch - cy_val) < 2000.0):
                bs['corner_ticks'] += 1

            # Per-step reward shaping
            center_prox = 1.0 - math.hypot(cx - cw_h, cy_val - ch_h) / _V2_HALF_DIAG
            r = 0.1 * center_prox * dt_scale   # small center-seeking bonus

            mass_delta = total_mass - bs['last_mass']
            if mass_delta >= 15.0:
                bs['cells_eaten_mass'] += mass_delta
                r += math.log1p(mass_delta) * dt_scale      # big reward for eating a cell
            elif 0 < mass_delta < 5.0:
                bs['food_eaten_count'] += 1
                r += math.log1p(mass_delta) * dt_scale      # small reward for eating food
            elif mass_delta <= -10.0:
                r -= math.log1p(-mass_delta) * dt_scale     # penalty for losing mass

            # Tiny survival bonus per tick
            r += 0.01 * dt_scale

            bs['last_mass'] = total_mass

            # Check if this bot just died (unregister called externally sets flag)
            if bs.get('died_this_tick', False):
                dones[i] = 1.0
                r -= 2.0   # death penalty

            rewards[i]    = r
            bot_alive[i]  = True
            bot_cx[i]     = cx
            bot_cy[i]     = cy_val
            bot_mass[i]   = total_mass
            bot_largest[i] = largest
            bot_scan[i]   = min(
                config.VIEW_BASE_SIZE / 2.0 * (max(total_mass, 20.0) / 100.0) ** config.VIEW_MASS_SCALE,
                3500.0,
            )
            intero[i, 0]  = math.log1p(total_mass / n_cells) / max_mass_log
            intero[i, 1]  = n_cells / _MAX_CELLS
            intero[i, 2]  = sr
            intero[i, 3]  = er
            intero[i, 4]  = max_merge / merge_max if max_merge < merge_max else 1.0
            last_vx[i]    = vx
            last_vy[i]    = vy

        # ---- Batched v2 perception ----
        obs_np = _build_inputs_batch_v2(
            world, plan_pids, bot_alive, bot_cx, bot_cy, bot_mass, bot_largest,
            bot_scan, intero, last_vx, last_vy, B,
        )  # (B, 122) float32

        # ---- Observation sanity check (periodic) ----
        if self.obs_check_interval > 0:
            self._obs_check_counter += 1
            if self._obs_check_counter >= self.obs_check_interval:
                self._obs_check_counter = 0
                check_obs(obs_np, label=f"step={self.step}")

        # ---- Policy inference ----
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            actions_t, logprobs_t, _, values_t = self.model.get_action_and_value(obs_t)

        # ---- Store rollout step ----
        if not self.inference_only:
            self.buffer.add(
                self.step,
                obs_np,
                actions_t,
                logprobs_t,
                rewards,
                dones,
                values_t,
            )
            self._last_obs   = obs_np
            self._last_dones = dones
            self.step       += 1

        # ---- Apply actions to game ----
        actions_np = actions_t.cpu().numpy()
        target_dist = _TARGET_DIST
        for i, pid in enumerate(plan_pids):
            if not bot_alive[i]:
                continue
            player = world.players.get(pid)
            if player is None:
                continue
            bs = self._state.get(pid)

            mx = float(math.tanh(actions_np[i, 0]))  # squash to [-1, 1]
            my = float(math.tanh(actions_np[i, 1]))
            mag = math.hypot(mx, my)
            if mag > 1e-6:
                mx /= mag
                my /= mag
            tx = max(0.0, min(float(cw), bot_cx[i] + mx * target_dist))
            ty = max(0.0, min(float(ch), bot_cy[i] + my * target_dist))
            player.target_x = tx
            player.target_y = ty

            sp = float(actions_np[i, 2]) > 0.5
            if sp and bs is not None:
                bs['last_split_tick'] = bs['ticks_alive']
            player.split_pending  = sp
            player.eject_pending  = float(actions_np[i, 3]) > 0.5

        # ---- Reset died_this_tick flags ----
        for pid in plan_pids:
            bs = self._state.get(pid)
            if bs is not None:
                bs['died_this_tick'] = False

    # ------------------------------------------------------------------
    # Rollout lifecycle
    # ------------------------------------------------------------------

    def is_rollout_full(self) -> bool:
        return self.step >= self.n_steps

    def finish_rollout(self, world: 'GameWorld') -> dict:
        """Bootstrap values for unfinished episodes and run the PPO update.

        Returns training stats dict from PPOTrainer.update().
        """
        if self._last_obs is None:
            return {}

        # Bootstrap values for still-alive bots
        obs_t = torch.as_tensor(self._last_obs, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            last_values = self.model.get_value(obs_t).squeeze(-1)  # (B,)

        last_dones = self._last_dones if self._last_dones is not None else np.zeros(len(self._pid_order))

        self.buffer.compute_returns_and_advantages(last_values, last_dones)
        stats = self.trainer.update(self.buffer)
        return stats

    def reset_rollout(self) -> None:
        """Reset step counter for the next rollout (does NOT clear bot states)."""
        self.step = 0
        self._last_obs   = None
        self._last_dones = None

    # ------------------------------------------------------------------
    # Fitness (for logging / leaderboard parity)
    # ------------------------------------------------------------------

    def _compute_fitness(self, bs: dict) -> float:
        time_alive = max(bs.get('game_time_alive', bs['ticks_alive'] * config.TICK_INTERVAL), 1e-6)
        ticks_alive = max(bs['ticks_alive'], 1)
        peak_mass  = bs['peak_mass']
        avg_mass   = bs['total_mass_accumulated'] / ticks_alive
        food_eaten = bs['food_eaten_count']
        cells_mass = bs.get('cells_eaten_mass', 0.0)
        edge_frac  = bs['edge_ticks'] / ticks_alive
        corner_frac = bs['corner_ticks'] / ticks_alive
        corner_penalty = edge_frac ** 2 + corner_frac ** 2
        survival_bonus = math.log1p(time_alive)
        fitness = (
            peak_mass * 0.4
            + avg_mass * 0.3
            + food_eaten * 3.0
            + cells_mass * 9.0
            + survival_bonus * 3.0
            - corner_penalty * 1000.0
        )
        return max(fitness, 0.0)

    def current_fitness(self, player_id: int) -> float:
        bs = self._state.get(player_id)
        return self._compute_fitness(bs) if bs else 0.0

    def get_pid_order(self) -> list[int]:
        return list(self._pid_order)
