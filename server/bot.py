from __future__ import annotations
import math
from typing import TYPE_CHECKING

import neat
import numpy as np
from scipy.spatial import cKDTree

from . import config
from .genetics import neat_config
from .nn_batch import build_batch_plan, BatchPlan

if TYPE_CHECKING:
    from .game import GameWorld


# ---------------------------------------------------------------------------
# Sensory constants
# ---------------------------------------------------------------------------

_SCAN_RADIUS   = 1500.0           # v1 only — fixed scan radius
_NUM_SECTORS   = 16               # directional bins around the bot
_SECTOR_ANGLE  = 2.0 * math.pi / _NUM_SECTORS
_TARGET_DIST   = 2000.0           # world units to project the move vector
_FOOD_MASS_NORM = 50.0            # food mass normalisation factor
_MAX_CELLS = config.MAX_CELLS
_CELL_POS_NORM = 1500.0
_SPLIT_TIME_NORM = 30.0
_VIRUS_RADIUS = config.VIRUS_RADIUS
_VEL_NORM = config.SPLIT_SPEED * config.TICK_INTERVAL  # max per-tick displacement (split burst)


class NullWebSocket:
    """Fake websocket for bots — silently discards all sends."""
    async def send(self, data) -> None:
        pass


class BotController:
    """
    Neural-network bot controller.
    Each bot's genome is a neat.DefaultGenome; a FeedForwardNetwork is derived
    from it once at registration and cached for the bot's lifetime.
    Fitness is tracked per-bot and reported on unregister (death/removal).
    """

    def __init__(self) -> None:
        # player_id -> bot state dict
        self._state: dict[int, dict] = {}
        self._plan: BatchPlan | None = None
        self._prev_plan: BatchPlan | None = None  # saved for hidden-state carry-over on rebuild
        # Species tracking: list of (species_id, representative_genome)
        self._species_reps: list[tuple[int, neat.DefaultGenome]] = []
        self._next_species_id: int = 0
        # Configurable fitness weights
        self._fw: dict[str, float] = {
            'peak':      0.4,
            'avg':       0.3,
            'food':      3.0,
            'cells':     9.0,
            'survival':  3.0,
            'corner':    1000.0,
            'death_exp': 3.0,
        }
        # Configurable Hebbian learning params
        self._hebbian_lr:    float = 0.004
        self._hebbian_decay: float = 0.92

    def reset_species(self) -> None:
        """Clear species representatives. Call at the start of each generation."""
        self._species_reps.clear()
        self._next_species_id = 0

    def register(self, player_id: int, start_x: float, start_y: float,
                 genome: neat.DefaultGenome) -> None:
        # Assign species by comparing against existing representative genomes
        threshold  = neat_config.species_set_config.compatibility_threshold
        species_id = -1
        for sid, rep in self._species_reps:
            if genome.distance(rep, neat_config.genome_config) < threshold:
                species_id = sid
                break
        if species_id == -1:
            species_id = self._next_species_id
            self._next_species_id += 1
            self._species_reps.append((species_id, genome))

        self._state[player_id] = {
            'genome':                  genome,
            'start_mass':              20.0,
            'ticks_alive':             0,
            'peak_mass':               20.0,
            'total_mass_accumulated':  20.0,
            'last_split_tick':         0,
            'deaths':                  0,
            'last_pos':                (start_x, start_y),
            'distance_traveled':       0.0,
            'last_mass':               20.0,
            'food_eaten_count':        0,
            'cells_eaten_mass':        0.0,
            'idle_ticks':              0,
            'edge_ticks':              0,
            'corner_ticks':            0,
            'species_id':              species_id,
            'last_vel':                (0.0, 0.0),
            'game_time_alive':         0.0,
        }
        if self._plan is not None:
            self._prev_plan = self._plan
        self._plan = None

    def unregister(self, player_id: int) -> tuple[neat.DefaultGenome | None, float]:
        """Remove bot state and return (genome, fitness). Returns (None, 0) if unknown."""
        bs = self._state.pop(player_id, None)
        # Write the dead bot's Hebbian weights back to its genome object so that
        # the genome saved in _gen_records carries the learned weights.
        # Use _plan if still live, _prev_plan if a prior death already cleared _plan.
        # Survivor write-backs are deferred to write_back_all() at generation end —
        # the plan rebuild carries Hebbian weights forward via prior_plan._W instead.
        active_plan = self._plan if self._plan is not None else self._prev_plan
        if active_plan is not None and bs is not None and bs.get('deaths', 0) == 0:
            for i, pid in enumerate(active_plan.player_ids):
                if pid == player_id:
                    active_plan.write_back_single(i, bs['genome'])
                    break
        if self._plan is not None:
            self._prev_plan = self._plan
            self._plan = None
        if bs is None:
            return None, 0.0
        return bs['genome'], self._compute_fitness(bs)

    def write_back_all(self) -> None:
        """Write current Hebbian weights into all alive bots' genome objects.
        Call once at generation end before collecting survivors."""
        if self._plan is None:
            return
        for i, pid in enumerate(self._plan.player_ids):
            bs = self._state.get(pid)
            if bs is not None:
                self._plan.write_back_single(i, bs['genome'])

    def current_fitness(self, player_id: int) -> float:
        """Return current fitness without removing bot state. Returns 0 if unknown."""
        bs = self._state.get(player_id)
        if bs is None:
            return 0.0
        return self._compute_fitness(bs)

    def apply_params(self, params: dict) -> None:
        """Apply a dict of training param overrides (called at generation start)."""
        fw = self._fw
        if 'fw_peak'       in params:
            fw['peak']           = float(params['fw_peak'])
        if 'fw_avg'        in params:
            fw['avg']            = float(params['fw_avg'])
        if 'fw_food'       in params:
            fw['food']           = float(params['fw_food'])
        if 'fw_cells'      in params:
            fw['cells']          = float(params['fw_cells'])
        if 'fw_survival'   in params:
            fw['survival']       = float(params['fw_survival'])
        if 'fw_corner'     in params:
            fw['corner']         = float(params['fw_corner'])
        if 'fw_death_exp'  in params:
            fw['death_exp']      = float(params['fw_death_exp'])
        if 'hebbian_lr'    in params:
            self._hebbian_lr     = float(params['hebbian_lr'])
        if 'hebbian_decay' in params:
            self._hebbian_decay  = float(params['hebbian_decay'])

    def _compute_fitness(self, bs: dict) -> float:
        time_alive = max(bs.get('game_time_alive', bs['ticks_alive'] * config.TICK_INTERVAL), 1e-6)
        ticks_alive = max(bs['ticks_alive'], 1)

        peak_mass        = bs['peak_mass']
        avg_mass         = bs['total_mass_accumulated'] / ticks_alive
        food_eaten       = bs['food_eaten_count']
        cells_eaten_mass = bs.get('cells_eaten_mass', 0.0)

        edge_fraction   = bs['edge_ticks']   / ticks_alive
        corner_fraction = bs['corner_ticks'] / ticks_alive
        corner_penalty  = (edge_fraction ** 2 + corner_fraction ** 2)

        survival_bonus = math.log1p(time_alive)
        fw = self._fw
        fitness = (
            peak_mass          * fw['peak']
            + avg_mass         * fw['avg']
            + food_eaten       * fw['food']
            + cells_eaten_mass * fw['cells']
            + survival_bonus   * fw['survival']
            - corner_penalty   * fw['corner']
        )

        deaths = bs.get('deaths', 0)
        if deaths > 0:
            fitness *= math.exp(-fw['death_exp'] * deaths)

        return max(fitness, 0.0)

    def get_genome(self, player_id: int) -> neat.DefaultGenome | None:
        bs = self._state.get(player_id)
        return bs['genome'] if bs else None

    # ------------------------------------------------------------------

    def update(self, world: 'GameWorld', dt: float) -> None:
        if not self._state:
            return

        # Rebuild batch plan when bot roster changes
        if self._plan is None:
            pids = [pid for pid in self._state if world.players.get(pid) is not None]
            if not pids:
                return
            genomes = [self._state[pid]['genome'] for pid in pids]
            self._plan = build_batch_plan(pids, genomes, neat_config, prior_plan=self._prev_plan, trace_decay=self._hebbian_decay)
            self._prev_plan = None

        if config.PERCEPTION_VERSION == 2:
            self._update_v2_batched(world, dt)
        else:
            self._update_legacy(world, dt)

    def _update_legacy(self, world: 'GameWorld', dt: float) -> None:
        """Per-bot perception fallback (used when PERCEPTION_VERSION != 2)."""
        dt_scale = dt / config.TICK_INTERVAL
        pending_rewards: list[tuple[int, float]] = []
        valid_pids:      list[int]               = []
        inputs_list:     list[list[float]]       = []

        for i, pid in enumerate(self._plan.player_ids):
            bs     = self._state.get(pid)
            player = world.players.get(pid)
            if bs is None or player is None:
                self._plan = None
                return
            bs['ticks_alive'] += 1
            bs['game_time_alive'] = bs.get('game_time_alive', 0.0) + dt
            total_mass = player.total_mass
            if total_mass > bs['peak_mass']:
                bs['peak_mass'] = total_mass
            bs['total_mass_accumulated'] += total_mass
            cx, cy = player.centroid
            last_x, last_y = bs['last_pos']
            vx, vy = cx - last_x, cy - last_y
            bs['last_vel'] = (vx, vy)
            dist_moved = math.hypot(vx, vy)
            bs['distance_traveled'] += dist_moved
            bs['last_pos'] = (cx, cy)
            if dist_moved < 1.5:
                bs['idle_ticks'] += 1
            _cw, _ch = config.WORLD_W, config.WORLD_H
            # Track edge/corner fractions for fitness stats
            if cx < 500.0 or cx > _cw - 500.0 or cy < 500.0 or cy > _ch - 500.0:
                bs['edge_ticks'] += 1
            if min(math.hypot(cx, cy), math.hypot(_cw - cx, cy),
                   math.hypot(cx, _ch - cy), math.hypot(_cw - cx, _ch - cy)) < 2000.0:
                bs['corner_ticks'] += 1
            # Center-proximity reward: 1.0 at center, 0.0 at corner
            _center_prox = 1.0 - math.hypot(cx - _cw * 0.5, cy - _ch * 0.5) / _V2_HALF_DIAG
            pending_rewards.append((i, 0.5 * _center_prox * dt_scale))
            mass_delta = total_mass - bs['last_mass']
            if mass_delta >= 15.0:
                bs['cells_eaten_mass'] += mass_delta
                pending_rewards.append((i, math.log1p(mass_delta) * dt_scale))
            elif 0 < mass_delta < 5.0:
                bs['food_eaten_count'] += 1
                pending_rewards.append((i, math.log1p(mass_delta) * dt_scale))
            elif mass_delta <= -10.0:
                pending_rewards.append((i, -math.log1p(-mass_delta) * dt_scale))
            bs['last_mass'] = total_mass
            inputs_list.append(_build_inputs(world, player, bs))
            valid_pids.append(pid)

        for plan_idx, reward in pending_rewards:
            self._plan.apply_reward(plan_idx, reward, lr=self._hebbian_lr)
        outputs_arr = self._plan.run(np.array(inputs_list, dtype=np.float32))
        for i, pid in enumerate(valid_pids):
            player = world.players[pid]
            bs = self._state[pid]
            mx, my = float(outputs_arr[i, 0]), float(outputs_arr[i, 1])
            mag = math.hypot(mx, my)
            if mag > 1e-6:
                mx /= mag; my /= mag
            cx, cy = player.centroid
            player.target_x = max(0.0, min(float(config.WORLD_W), cx + mx * _TARGET_DIST))
            player.target_y = max(0.0, min(float(config.WORLD_H), cy + my * _TARGET_DIST))
            sp = float(outputs_arr[i, 2]) > 0.0
            if sp:
                bs['last_split_tick'] = bs['ticks_alive']
            player.split_pending = sp
            player.eject_pending = float(outputs_arr[i, 3]) > 0.0

    def _update_v2_batched(self, world: 'GameWorld', dt: float) -> None:
        """Vectorized v2 perception across all bots in one numpy pass."""
        dt_scale = dt / config.TICK_INTERVAL
        plan_pids = self._plan.player_ids
        B = len(plan_pids)
        players_map = world.players
        states = self._state

        # ---- Phase 1: per-bot snapshot + fitness/reward tracking ----
        bot_alive   = np.zeros(B, dtype=bool)
        bot_cx      = np.zeros(B, dtype=np.float64)
        bot_cy      = np.zeros(B, dtype=np.float64)
        bot_mass    = np.zeros(B, dtype=np.float64)
        bot_largest = np.zeros(B, dtype=np.float64)
        bot_scan    = np.zeros(B, dtype=np.float64)
        intero      = np.zeros((B, 5), dtype=np.float64)
        last_vx     = np.zeros(B, dtype=np.float64)
        last_vy     = np.zeros(B, dtype=np.float64)
        pending_rewards: list[tuple[int, float]] = []

        cw = config.WORLD_W
        ch = config.WORLD_H
        cw_h = cw * 0.5
        ch_h = ch * 0.5
        min_split = config.MIN_SPLIT_MASS
        min_eject = config.EJECT_MIN_MASS
        max_mass_log = _V2_MAX_MASS_LOG
        merge_max = _V2_MERGE_TIMER_MAX

        for i, pid in enumerate(plan_pids):
            bs = states.get(pid)
            player = players_map.get(pid)
            if bs is None or player is None:
                self._plan = None
                return
            cells = player.cells
            if not cells:
                continue

            # One-pass aggregation: total_mass, centroid, largest, max merge, flags.
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
            if total_mass > 0.0:
                cx = wx / total_mass
                cy = wy / total_mass
            else:
                cx = cw_h
                cy = ch_h

            bs['ticks_alive'] += 1
            bs['game_time_alive'] = bs.get('game_time_alive', 0.0) + dt
            if total_mass > bs['peak_mass']:
                bs['peak_mass'] = total_mass
            bs['total_mass_accumulated'] += total_mass
            last_x, last_y = bs['last_pos']
            vx = cx - last_x
            vy = cy - last_y
            bs['last_vel'] = (vx, vy)
            dist_moved = math.hypot(vx, vy)
            bs['distance_traveled'] += dist_moved
            bs['last_pos'] = (cx, cy)

            if dist_moved < 1.5:
                bs['idle_ticks'] += 1
            # Track edge/corner fractions for fitness stats
            if cx < 500.0 or cx > cw - 500.0 or cy < 500.0 or cy > ch - 500.0:
                bs['edge_ticks'] += 1
            if (math.hypot(cx, cy) < 2000.0 or math.hypot(cw - cx, cy) < 2000.0
                    or math.hypot(cx, ch - cy) < 2000.0
                    or math.hypot(cw - cx, ch - cy) < 2000.0):
                bs['corner_ticks'] += 1
            # Center-proximity reward: 1.0 at center, 0.0 at corner
            _center_prox = 1.0 - math.hypot(cx - cw_h, cy - ch_h) / _V2_HALF_DIAG
            pending_rewards.append((i, 0.5 * _center_prox * dt_scale))

            mass_delta = total_mass - bs['last_mass']
            if mass_delta >= 15.0:
                bs['cells_eaten_mass'] += mass_delta
                pending_rewards.append((i, math.log1p(mass_delta) * dt_scale))
            elif 0 < mass_delta < 5.0:
                bs['food_eaten_count'] += 1
                pending_rewards.append((i, math.log1p(mass_delta) * dt_scale))
            elif mass_delta <= -10.0:
                pending_rewards.append((i, -math.log1p(-mass_delta) * dt_scale))
            bs['last_mass'] = total_mass

            # Snapshot for batched perception
            bot_alive[i] = True
            bot_cx[i] = cx
            bot_cy[i] = cy
            bot_mass[i] = total_mass
            bot_largest[i] = largest
            bot_scan[i] = min(
                config.VIEW_BASE_SIZE / 2.0 * (max(total_mass, 20.0) / 100.0) ** config.VIEW_MASS_SCALE,
                3500.0,
            )
            intero[i, 0] = math.log1p(total_mass / n_cells) / max_mass_log
            intero[i, 1] = n_cells / _MAX_CELLS
            intero[i, 2] = sr
            intero[i, 3] = er
            intero[i, 4] = max_merge / merge_max if max_merge < merge_max else 1.0
            last_vx[i] = vx
            last_vy[i] = vy

        # ---- Phase 2: batched perception ----
        inputs_batch = _build_inputs_batch_v2(
            world, plan_pids, bot_alive, bot_cx, bot_cy, bot_mass, bot_largest,
            bot_scan, intero, last_vx, last_vy, B,
        )

        # ---- Phase 3: rewards ----
        for plan_idx, reward in pending_rewards:
            self._plan.apply_reward(plan_idx, reward, lr=self._hebbian_lr)

        # ---- Phase 4: network forward pass ----
        outputs_arr = self._plan.run(inputs_batch)

        # ---- Phase 5: apply outputs ----
        target_dist = _TARGET_DIST
        for i, pid in enumerate(plan_pids):
            if not bot_alive[i]:
                continue
            player = players_map.get(pid)
            if player is None:
                continue
            bs = states.get(pid)
            mx = float(outputs_arr[i, 0])
            my = float(outputs_arr[i, 1])
            mag = math.hypot(mx, my)
            if mag > 1e-6:
                mx /= mag
                my /= mag
            tx = bot_cx[i] + mx * target_dist
            ty = bot_cy[i] + my * target_dist
            if tx < 0.0:    tx = 0.0
            elif tx > cw:   tx = cw
            if ty < 0.0:    ty = 0.0
            elif ty > ch:   ty = ch
            player.target_x = tx
            player.target_y = ty
            sp = float(outputs_arr[i, 2]) > 0.0
            if sp and bs is not None:
                bs['last_split_tick'] = bs['ticks_alive']
            player.split_pending = sp
            player.eject_pending = float(outputs_arr[i, 3]) > 0.0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _build_inputs(world: 'GameWorld', player, bot_state: dict) -> list[float]:
    if config.PERCEPTION_VERSION == 2:
        return _build_inputs_v2(world, player, bot_state)
    return _build_inputs_v1(world, player, bot_state)


# ---------------------------------------------------------------------------
# Perception v1 — 16-sector spatial encoding (181 inputs)
# ---------------------------------------------------------------------------
#
# Layout: 16 sectors × 8 values = 128, + 16 cell masses, + 16 × 2 cell
# positions = 32, + 1 split timer, + 4 wall proximities  →  181 inputs total.
#
# Per sector:
#   [0] food_proximity   — 1 − dist/SCAN_RADIUS
#   [1] food_mass_norm   — food.mass / _FOOD_MASS_NORM, clamped 0–1
#   [2] prey_proximity   — 1 − dist/SCAN_RADIUS  (nearest smaller cell)
#   [3] prey_smallness   — 1 − prey_mass/own_mass
#   [4] threat_proximity — 1 − dist/SCAN_RADIUS  (nearest larger cell)
#   [5] threat_danger    — 1 − own_mass/threat_mass
#   [6] virus_proximity  — 1 − dist/SCAN_RADIUS
#   [7] virus_danger     — 1.0 if any cell > VIRUS_SPLIT_THRESHOLD
#
# Wall proximities (0 = far, 1 = at wall): left, right, top, bottom

def _build_inputs_v1(world: 'GameWorld', player, bot_state: dict) -> list[float]:
    if not player.cells:
        return [0.0] * (_NUM_SECTORS * 8 + _MAX_CELLS * 3 + 1 + 4)

    cx, cy = player.centroid
    own_mass = player.total_mass
    sorted_cells = sorted(player.cells, key=lambda c: c.mass, reverse=True)
    largest_cell_mass = sorted_cells[0].mass

    sectors          = [[0.0] * 8 for _ in range(_NUM_SECTORS)]
    food_best_dist   = [_SCAN_RADIUS] * _NUM_SECTORS
    prey_best_dist   = [_SCAN_RADIUS] * _NUM_SECTORS
    threat_best_dist = [_SCAN_RADIUS] * _NUM_SECTORS
    virus_best_dist  = [_SCAN_RADIUS] * _NUM_SECTORS

    for fid in world.food_grid.query_radius(cx, cy, _SCAN_RADIUS):
        food = world.food_mgr.get(fid)
        if food is None:
            continue
        dx, dy = food.x - cx, food.y - cy
        dist = math.hypot(dx, dy)
        if dist < 1e-6 or dist >= _SCAN_RADIUS:
            continue
        s = _sector_v1(dx, dy)
        if dist < food_best_dist[s]:
            food_best_dist[s] = dist
            sectors[s][0] = 1.0 - dist / _SCAN_RADIUS
            sectors[s][1] = min(food.mass / _FOOD_MASS_NORM, 1.0)

    for cid in world.cell_grid.query_radius(cx, cy, _SCAN_RADIUS):
        entry = world.cell_map.get(cid)
        if entry is None:
            continue
        other_cell, other_player = entry
        if other_player.id == player.id:
            continue
        dx, dy = other_cell.x - cx, other_cell.y - cy
        dist = math.hypot(dx, dy)
        if dist < 1e-6 or dist >= _SCAN_RADIUS:
            continue
        s = _sector_v1(dx, dy)
        ocm = other_cell.mass
        if ocm * config.EAT_RATIO <= largest_cell_mass:
            if dist < prey_best_dist[s]:
                prey_best_dist[s] = dist
                sectors[s][2] = 1.0 - dist / _SCAN_RADIUS
                sectors[s][3] = 1.0 - ocm / largest_cell_mass
        elif ocm >= largest_cell_mass * config.EAT_RATIO:
            if dist < threat_best_dist[s]:
                threat_best_dist[s] = dist
                sectors[s][4] = 1.0 - dist / _SCAN_RADIUS
                sectors[s][5] = 1.0 - min(largest_cell_mass / ocm, 1.0)

    virus_danger_val = 1.0 if sorted_cells[0].mass > config.VIRUS_SPLIT_THRESHOLD else 0.0
    if world.virus_grid is not None:
        for vid in world.virus_grid.query_radius(cx, cy, _SCAN_RADIUS):
            virus = world.virus_mgr.get(vid)
            if virus is None:
                continue
            dx, dy = virus.x - cx, virus.y - cy
            dist = math.hypot(dx, dy)
            if dist < 1e-6 or dist >= _SCAN_RADIUS:
                continue
            s = _sector_v1(dx, dy)
            if dist < virus_best_dist[s]:
                virus_best_dist[s] = dist
                sectors[s][6] = 1.0 - dist / _SCAN_RADIUS
                sectors[s][7] = virus_danger_val

    inputs = [v for s in sectors for v in s]

    own_mass_safe = max(own_mass, 1e-6)
    for i in range(_MAX_CELLS):
        inputs.append(sorted_cells[i].mass / own_mass_safe if i < len(sorted_cells) else 0.0)
    for i in range(_MAX_CELLS):
        if i < len(sorted_cells):
            c = sorted_cells[i]
            inputs.append(max(-1.0, min(1.0, (c.x - cx) / _CELL_POS_NORM)))
            inputs.append(max(-1.0, min(1.0, (c.y - cy) / _CELL_POS_NORM)))
        else:
            inputs.append(0.0)
            inputs.append(0.0)

    ticks_alive = bot_state['ticks_alive']
    last_split_tick = bot_state['last_split_tick']
    time_since_split = max(ticks_alive - last_split_tick, 0) * config.TICK_INTERVAL
    inputs.append(min(time_since_split / _SPLIT_TIME_NORM, 1.0))

    # Wall proximities — 0 = far, 1 = at wall (within SCAN_RADIUS)
    _cw, _ch = config.WORLD_W, config.WORLD_H
    inputs.append(max(0.0, 1.0 - cx            / _SCAN_RADIUS))  # left
    inputs.append(max(0.0, 1.0 - (_cw - cx)    / _SCAN_RADIUS))  # right
    inputs.append(max(0.0, 1.0 - cy            / _SCAN_RADIUS))  # top
    inputs.append(max(0.0, 1.0 - (_ch - cy)    / _SCAN_RADIUS))  # bottom

    return inputs


def _sector_v1(dx: float, dy: float) -> int:
    angle = math.atan2(dy, dx) % (2.0 * math.pi)
    return int(angle / _SECTOR_ANGLE) % _NUM_SECTORS


# ---------------------------------------------------------------------------
# Perception v2 — raycast + quadrant model (58 inputs)
# ---------------------------------------------------------------------------
#
# Layout:
#   5   interoceptive   (avg_mass_per_cell, num_cells, split/eject ready, merge timer)
#   40  raycasts        (8 rays × 5: food, prey, threat, virus, wall proximity)
#   8   quadrant sums   (4 quadrants × food_density + threat_level)
#   3   contextual      (center dist, mass rank, danger score)
#   2   velocity        (vx, vy normalized by split-burst speed)
#   ─── total: 58
#
# Scan radius scales with mass like the player viewport — larger bots see
# further, matching their slower speed and greater reaction-time need.
# Ray encoding: 0.0 = nothing in range / far, 1.0 = object at bot centroid.
# Higher activation always means "pay attention here."

_V2_NUM_RAYS        = 8
_V2_RAY_ANGLE       = 2.0 * math.pi / _V2_NUM_RAYS
_V2_MAX_MASS_LOG    = math.log1p(5000.0)          # log-scale ceiling for mass
_V2_MERGE_TIMER_MAX = config.MERGE_TIME_BASE + config.MERGE_TIME_MASS_FACTOR * 5000.0
_V2_QUAD_FOOD_NORM  = 20.0    # expected max food pellets per quadrant in scan area
_V2_QUAD_THREAT_NORM = 5000.0 # expected max threat mass per quadrant
_V2_DANGER_NORM     = 10.0    # normalization divisor for danger_score
_V2_HALF_DIAG       = math.hypot(config.WORLD_W / 2.0, config.WORLD_H / 2.0)
_V2_TWO_PI          = 2.0 * math.pi
_V2_INV_RAY_ANGLE   = 1.0 / _V2_RAY_ANGLE


def _build_inputs_v2(world: 'GameWorld', player, bot_state: dict) -> list[float]:
    if not player.cells:
        return [0.0] * 74

    cx, cy = player.centroid
    own_mass = player.total_mass
    cells = player.cells
    largest_cell_mass = max(c.mass for c in cells)

    # Scan radius mirrors the player viewport scale — bigger bots see further
    scan_radius = min(
        config.VIEW_BASE_SIZE / 2.0 * (max(own_mass, 20.0) / 100.0) ** config.VIEW_MASS_SCALE,
        3500.0,
    )
    scan_sq  = scan_radius * scan_radius
    inv_scan = 1.0 / scan_radius
    eat_ratio = config.EAT_RATIO

    # --- Interoceptive (5) ---
    n_cells = len(cells)
    avg_mass_n    = math.log1p(own_mass / n_cells) / _V2_MAX_MASS_LOG
    num_cells_n   = n_cells / _MAX_CELLS
    split_ready   = 0.0
    eject_ready   = 0.0
    max_merge     = 0.0
    for c in cells:
        m = c.mass
        if m >= config.MIN_SPLIT_MASS:
            split_ready = 1.0
        if m >= config.EJECT_MIN_MASS:
            eject_ready = 1.0
        if c.merge_timer > max_merge:
            max_merge = c.merge_timer
    merge_timer_n = min(max_merge / _V2_MERGE_TIMER_MAX, 1.0)

    # --- Raycast & quadrant accumulators (Python floats — small per-bot loops) ---
    ray_food_0 = ray_food_1 = ray_food_2 = ray_food_3 = 0.0
    ray_food_4 = ray_food_5 = ray_food_6 = ray_food_7 = 0.0
    ray_prey_0 = ray_prey_1 = ray_prey_2 = ray_prey_3 = 0.0
    ray_prey_4 = ray_prey_5 = ray_prey_6 = ray_prey_7 = 0.0
    ray_prey_mass_0 = ray_prey_mass_1 = ray_prey_mass_2 = ray_prey_mass_3 = 0.0
    ray_prey_mass_4 = ray_prey_mass_5 = ray_prey_mass_6 = ray_prey_mass_7 = 0.0
    ray_threat_0 = ray_threat_1 = ray_threat_2 = ray_threat_3 = 0.0
    ray_threat_4 = ray_threat_5 = ray_threat_6 = ray_threat_7 = 0.0
    ray_threat_mass_0 = ray_threat_mass_1 = ray_threat_mass_2 = ray_threat_mass_3 = 0.0
    ray_threat_mass_4 = ray_threat_mass_5 = ray_threat_mass_6 = ray_threat_mass_7 = 0.0
    ray_virus_0 = ray_virus_1 = ray_virus_2 = ray_virus_3 = 0.0
    ray_virus_4 = ray_virus_5 = ray_virus_6 = ray_virus_7 = 0.0
    quad_food_0 = quad_food_1 = quad_food_2 = quad_food_3 = 0.0
    quad_threat_0 = quad_threat_1 = quad_threat_2 = quad_threat_3 = 0.0
    visible_count = 0
    prey_count    = 0
    danger_sum    = 0.0
    own_inv       = 1.0 / max(own_mass, 1e-6)

    # --- Food (inline, sq-distance early out) ---
    food_get   = world.food_mgr._food.__getitem__
    food_dict  = world.food_mgr._food
    inv_ray_a  = _V2_INV_RAY_ANGLE
    two_pi     = _V2_TWO_PI
    n_rays     = _V2_NUM_RAYS

    for fid in world.food_grid.query_radius(cx, cy, scan_radius):
        food = food_dict.get(fid)
        if food is None:
            continue
        dx = food.x - cx
        dy = food.y - cy
        d2 = dx * dx + dy * dy
        if d2 < 1e-12 or d2 >= scan_sq:
            continue
        dist = math.sqrt(d2)
        dn = 1.0 - dist * inv_scan
        ang = math.atan2(dy, dx)
        if ang < 0.0:
            ang += two_pi
        s = int(ang * inv_ray_a) % n_rays
        # Inline ray max
        if s == 0:
            if dn > ray_food_0: ray_food_0 = dn
        elif s == 1:
            if dn > ray_food_1: ray_food_1 = dn
        elif s == 2:
            if dn > ray_food_2: ray_food_2 = dn
        elif s == 3:
            if dn > ray_food_3: ray_food_3 = dn
        elif s == 4:
            if dn > ray_food_4: ray_food_4 = dn
        elif s == 5:
            if dn > ray_food_5: ray_food_5 = dn
        elif s == 6:
            if dn > ray_food_6: ray_food_6 = dn
        else:
            if dn > ray_food_7: ray_food_7 = dn
        # Inline quadrant: right=dx>=0, down=dy>=0
        if dx >= 0:
            if dy >= 0: quad_food_3 += 1.0
            else:        quad_food_0 += 1.0
        else:
            if dy >= 0: quad_food_2 += 1.0
            else:        quad_food_1 += 1.0

    # --- Enemy cells (inline) ---
    own_pid = player.id
    cell_map_get = world.cell_map.get
    threat_thresh = largest_cell_mass * eat_ratio
    prey_thresh   = largest_cell_mass / eat_ratio
    for cid in world.cell_grid.query_radius(cx, cy, scan_radius):
        entry = cell_map_get(cid)
        if entry is None:
            continue
        other_cell, other_player = entry
        if other_player.id == own_pid:
            continue
        dx = other_cell.x - cx
        dy = other_cell.y - cy
        d2 = dx * dx + dy * dy
        if d2 < 1e-12 or d2 >= scan_sq:
            continue
        dist = math.sqrt(d2)
        ocm = other_cell.mass
        dn = 1.0 - dist * inv_scan
        ang = math.atan2(dy, dx)
        if ang < 0.0:
            ang += two_pi
        s = int(ang * inv_ray_a) % n_rays
        visible_count += 1
        if ocm <= prey_thresh:
            prey_count += 1
            pm = ocm * own_inv  # prey mass / own mass [0, ~0.8]
            if s == 0:
                if dn > ray_prey_0: ray_prey_0 = dn; ray_prey_mass_0 = pm
            elif s == 1:
                if dn > ray_prey_1: ray_prey_1 = dn; ray_prey_mass_1 = pm
            elif s == 2:
                if dn > ray_prey_2: ray_prey_2 = dn; ray_prey_mass_2 = pm
            elif s == 3:
                if dn > ray_prey_3: ray_prey_3 = dn; ray_prey_mass_3 = pm
            elif s == 4:
                if dn > ray_prey_4: ray_prey_4 = dn; ray_prey_mass_4 = pm
            elif s == 5:
                if dn > ray_prey_5: ray_prey_5 = dn; ray_prey_mass_5 = pm
            elif s == 6:
                if dn > ray_prey_6: ray_prey_6 = dn; ray_prey_mass_6 = pm
            else:
                if dn > ray_prey_7: ray_prey_7 = dn; ray_prey_mass_7 = pm
        elif ocm >= threat_thresh:
            tm = own_mass / max(ocm, 1e-6)  # own/threat [0,1]: 1=borderline, ~0=huge threat
            if tm > 1.0: tm = 1.0
            if s == 0:
                if dn > ray_threat_0: ray_threat_0 = dn; ray_threat_mass_0 = tm
            elif s == 1:
                if dn > ray_threat_1: ray_threat_1 = dn; ray_threat_mass_1 = tm
            elif s == 2:
                if dn > ray_threat_2: ray_threat_2 = dn; ray_threat_mass_2 = tm
            elif s == 3:
                if dn > ray_threat_3: ray_threat_3 = dn; ray_threat_mass_3 = tm
            elif s == 4:
                if dn > ray_threat_4: ray_threat_4 = dn; ray_threat_mass_4 = tm
            elif s == 5:
                if dn > ray_threat_5: ray_threat_5 = dn; ray_threat_mass_5 = tm
            elif s == 6:
                if dn > ray_threat_6: ray_threat_6 = dn; ray_threat_mass_6 = tm
            else:
                if dn > ray_threat_7: ray_threat_7 = dn; ray_threat_mass_7 = tm
            if dx >= 0:
                if dy >= 0: quad_threat_3 += ocm
                else:        quad_threat_0 += ocm
            else:
                if dy >= 0: quad_threat_2 += ocm
                else:        quad_threat_1 += ocm
            danger_sum += (ocm * own_inv) * (1.0 - dn)

    # --- Viruses (inline) ---
    if world.virus_grid is not None:
        virus_dict = world.virus_mgr._viruses
        for vid in world.virus_grid.query_radius(cx, cy, scan_radius):
            virus = virus_dict.get(vid)
            if virus is None:
                continue
            dx = virus.x - cx
            dy = virus.y - cy
            d2 = dx * dx + dy * dy
            if d2 < 1e-12 or d2 >= scan_sq:
                continue
            dist = math.sqrt(d2)
            dn = 1.0 - dist * inv_scan
            ang = math.atan2(dy, dx)
            if ang < 0.0:
                ang += two_pi
            s = int(ang * inv_ray_a) % n_rays
            if s == 0:
                if dn > ray_virus_0: ray_virus_0 = dn
            elif s == 1:
                if dn > ray_virus_1: ray_virus_1 = dn
            elif s == 2:
                if dn > ray_virus_2: ray_virus_2 = dn
            elif s == 3:
                if dn > ray_virus_3: ray_virus_3 = dn
            elif s == 4:
                if dn > ray_virus_4: ray_virus_4 = dn
            elif s == 5:
                if dn > ray_virus_5: ray_virus_5 = dn
            elif s == 6:
                if dn > ray_virus_6: ray_virus_6 = dn
            else:
                if dn > ray_virus_7: ray_virus_7 = dn

    # --- Wall distances (analytical) ---
    world_w = config.WORLD_W
    world_h = config.WORLD_H
    rw_0 = max(0.0, 1.0 - ((world_w - cx) / max(_V2_WALL_RDX_TBL[0], 1e-9)) * inv_scan)
    rw_1 = max(0.0, 1.0 - _wall_dist(cx, cy, _V2_WALL_RDX_TBL[1], _V2_WALL_RDY_TBL[1], world_w, world_h) * inv_scan)
    rw_2 = max(0.0, 1.0 - _wall_dist(cx, cy, _V2_WALL_RDX_TBL[2], _V2_WALL_RDY_TBL[2], world_w, world_h) * inv_scan)
    rw_3 = max(0.0, 1.0 - _wall_dist(cx, cy, _V2_WALL_RDX_TBL[3], _V2_WALL_RDY_TBL[3], world_w, world_h) * inv_scan)
    rw_4 = max(0.0, 1.0 - (cx / max(-_V2_WALL_RDX_TBL[4], 1e-9)) * inv_scan)
    rw_5 = max(0.0, 1.0 - _wall_dist(cx, cy, _V2_WALL_RDX_TBL[5], _V2_WALL_RDY_TBL[5], world_w, world_h) * inv_scan)
    rw_6 = max(0.0, 1.0 - _wall_dist(cx, cy, _V2_WALL_RDX_TBL[6], _V2_WALL_RDY_TBL[6], world_w, world_h) * inv_scan)
    rw_7 = max(0.0, 1.0 - _wall_dist(cx, cy, _V2_WALL_RDX_TBL[7], _V2_WALL_RDY_TBL[7], world_w, world_h) * inv_scan)

    # --- Quadrant summaries (8) ---
    qfn = _V2_QUAD_FOOD_NORM
    qtn = _V2_QUAD_THREAT_NORM
    qf_0 = quad_food_0 / qfn if quad_food_0 < qfn else 1.0
    qf_1 = quad_food_1 / qfn if quad_food_1 < qfn else 1.0
    qf_2 = quad_food_2 / qfn if quad_food_2 < qfn else 1.0
    qf_3 = quad_food_3 / qfn if quad_food_3 < qfn else 1.0
    qt_0 = quad_threat_0 / qtn if quad_threat_0 < qtn else 1.0
    qt_1 = quad_threat_1 / qtn if quad_threat_1 < qtn else 1.0
    qt_2 = quad_threat_2 / qtn if quad_threat_2 < qtn else 1.0
    qt_3 = quad_threat_3 / qtn if quad_threat_3 < qtn else 1.0

    # --- Contextual (3) ---
    cdx = cx - world_w * 0.5
    cdy = cy - world_h * 0.5
    center_dist = math.sqrt(cdx * cdx + cdy * cdy) / _V2_HALF_DIAG
    if center_dist > 1.0:
        center_dist = 1.0
    mass_rank = (prey_count / visible_count) if visible_count else 0.5
    danger_score = danger_sum / _V2_DANGER_NORM
    if danger_score > 1.0:
        danger_score = 1.0

    # --- Velocity (2) ---
    vx, vy = bot_state.get('last_vel', (0.0, 0.0))
    vel_x = vx / _VEL_NORM
    vel_y = vy / _VEL_NORM
    if vel_x > 1.0: vel_x = 1.0
    elif vel_x < -1.0: vel_x = -1.0
    if vel_y > 1.0: vel_y = 1.0
    elif vel_y < -1.0: vel_y = -1.0

    return [
        avg_mass_n, num_cells_n, split_ready, eject_ready, merge_timer_n,
        ray_food_0, ray_prey_0, ray_prey_mass_0, ray_threat_0, ray_threat_mass_0, ray_virus_0, rw_0,
        ray_food_1, ray_prey_1, ray_prey_mass_1, ray_threat_1, ray_threat_mass_1, ray_virus_1, rw_1,
        ray_food_2, ray_prey_2, ray_prey_mass_2, ray_threat_2, ray_threat_mass_2, ray_virus_2, rw_2,
        ray_food_3, ray_prey_3, ray_prey_mass_3, ray_threat_3, ray_threat_mass_3, ray_virus_3, rw_3,
        ray_food_4, ray_prey_4, ray_prey_mass_4, ray_threat_4, ray_threat_mass_4, ray_virus_4, rw_4,
        ray_food_5, ray_prey_5, ray_prey_mass_5, ray_threat_5, ray_threat_mass_5, ray_virus_5, rw_5,
        ray_food_6, ray_prey_6, ray_prey_mass_6, ray_threat_6, ray_threat_mass_6, ray_virus_6, rw_6,
        ray_food_7, ray_prey_7, ray_prey_mass_7, ray_threat_7, ray_threat_mass_7, ray_virus_7, rw_7,
        qf_0, qf_1, qf_2, qf_3,
        qt_0, qt_1, qt_2, qt_3,
        center_dist, mass_rank, danger_score, vel_x, vel_y,
    ]


_V2_WALL_RDX_TBL = tuple(math.cos(i * _V2_RAY_ANGLE) for i in range(_V2_NUM_RAYS))
_V2_WALL_RDY_TBL = tuple(math.sin(i * _V2_RAY_ANGLE) for i in range(_V2_NUM_RAYS))


def _wall_dist(cx: float, cy: float, rdx: float, rdy: float, world_w: float, world_h: float) -> float:
    """Distance along ray (rdx, rdy) until it hits the world rect."""
    t = 1e18
    if rdx > 1e-9:
        t = (world_w - cx) / rdx
    elif rdx < -1e-9:
        t = cx / -rdx
    if rdy > 1e-9:
        t2 = (world_h - cy) / rdy
        if t2 < t: t = t2
    elif rdy < -1e-9:
        t2 = cy / -rdy
        if t2 < t: t = t2
    return t


def _sector_v2(dx: float, dy: float) -> int:
    angle = math.atan2(dy, dx) % _V2_TWO_PI
    return int(angle * _V2_INV_RAY_ANGLE) % _V2_NUM_RAYS


def _quadrant(dx: float, dy: float) -> int:
    """0=right+up, 1=left+up, 2=left+down, 3=right+down (screen-space y)."""
    right = dx >= 0
    down  = dy >= 0
    if right and not down:
        return 0
    if not right and not down:
        return 1
    if not right and down:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Batched perception (v2) — process all bots in a single numpy pass
# ---------------------------------------------------------------------------
#
# Per-bot vectorisation hits a wall around 100–200 items/bot: numpy's
# per-call dispatch overhead (~5–20 μs) × ~10 ops × B bots dominates.
# Concatenating all (bot, item) pairs into one edge list lets one numpy
# operation amortise the overhead across the entire population. With
# B = 400 bots and ~150 items/bot, the food edge list has ~60K rows —
# squarely in numpy's sweet spot. Reductions back to per-bot arrays use
# np.maximum.at / np.add.at on (bot_idx, sector) tuples.

_V2_WALL_RDX_ARR = np.array(
    [math.cos(i * _V2_RAY_ANGLE) for i in range(_V2_NUM_RAYS)], dtype=np.float64,
)
_V2_WALL_RDY_ARR = np.array(
    [math.sin(i * _V2_RAY_ANGLE) for i in range(_V2_NUM_RAYS)], dtype=np.float64,
)
_V2_INV_VEL_NORM = 1.0 / _VEL_NORM
_V2_INV_DANGER   = 1.0 / _V2_DANGER_NORM
_V2_INV_QFOOD    = 1.0 / _V2_QUAD_FOOD_NORM
_V2_INV_QTHREAT  = 1.0 / _V2_QUAD_THREAT_NORM
_V2_INV_HALF_DIAG = 1.0 / _V2_HALF_DIAG


def _build_inputs_batch_v2(
    world,
    plan_pids: list[int],
    bot_alive: np.ndarray,
    bot_cx: np.ndarray,
    bot_cy: np.ndarray,
    bot_mass: np.ndarray,
    bot_largest: np.ndarray,
    bot_scan: np.ndarray,
    intero: np.ndarray,
    last_vx: np.ndarray,
    last_vy: np.ndarray,
    B: int,
) -> np.ndarray:
    """All-bots batched v2 perception.
    Returns shape (B, 74) float32 inputs ready for plan.run().

    Strategy: all-pairs distance in numpy. We snapshot all food / cell /
    virus positions into numpy arrays once per tick, then compute the full
    (B, N) distance matrix and mask in vectorised form. This bypasses the
    spatial grid for perception entirely — the grid is still used for
    physics-side queries which are O(per-cell) and too small to amortise
    numpy overhead. Memory: B × F × 8 ≈ 20 MB scratch for B=400, F=6000."""

    # ---- Snapshot positions into numpy (one Python pass per manager) ----
    food_dict = world.food_mgr._food
    n_food = len(food_dict)
    if n_food:
        food_x = np.empty(n_food, dtype=np.float64)
        food_y = np.empty(n_food, dtype=np.float64)
        i = 0
        for f in food_dict.values():
            food_x[i] = f.x
            food_y[i] = f.y
            i += 1
    else:
        food_x = np.empty(0, dtype=np.float64)
        food_y = np.empty(0, dtype=np.float64)

    cell_map = world.cell_map
    n_cells = len(cell_map)
    if n_cells:
        cell_x = np.empty(n_cells, dtype=np.float64)
        cell_y = np.empty(n_cells, dtype=np.float64)
        cell_m = np.empty(n_cells, dtype=np.float64)
        cell_pid = np.empty(n_cells, dtype=np.int64)
        i = 0
        for cell, owner in cell_map.values():
            cell_x[i] = cell.x
            cell_y[i] = cell.y
            cell_m[i] = cell.mass
            cell_pid[i] = owner.id
            i += 1
    else:
        cell_x = np.empty(0, dtype=np.float64)
        cell_y = np.empty(0, dtype=np.float64)
        cell_m = np.empty(0, dtype=np.float64)
        cell_pid = np.empty(0, dtype=np.int64)

    if world.virus_grid is not None:
        virus_dict = world.virus_mgr._viruses
        n_v = len(virus_dict)
        if n_v:
            virus_x = np.empty(n_v, dtype=np.float64)
            virus_y = np.empty(n_v, dtype=np.float64)
            i = 0
            for v in virus_dict.values():
                virus_x[i] = v.x
                virus_y[i] = v.y
                i += 1
        else:
            virus_x = np.empty(0, dtype=np.float64)
            virus_y = np.empty(0, dtype=np.float64)
    else:
        virus_x = np.empty(0, dtype=np.float64)
        virus_y = np.empty(0, dtype=np.float64)

    bot_pid_arr = np.asarray(plan_pids, dtype=np.int64)

    # ---- Food: KDTree-based spatial reduction ----
    if n_food:
        ray_food, quad_food = _allpairs_food(
            food_x, food_y, bot_cx, bot_cy, bot_scan, bot_alive, B,
        )
    else:
        ray_food  = np.zeros((B, _V2_NUM_RAYS), dtype=np.float32)
        quad_food = np.zeros((B, 4),            dtype=np.float32)

    # ---- Cells: all-pairs reduction (excludes same-player pairs) ----
    if n_cells:
        ray_prey, ray_prey_mass, ray_threat, ray_threat_mass, quad_threat, mass_rank, danger_score = _allpairs_cells(
            cell_x, cell_y, cell_m, cell_pid,
            bot_cx, bot_cy, bot_scan, bot_pid_arr, bot_largest, bot_mass,
            bot_alive, B,
        )
    else:
        ray_prey      = np.zeros((B, _V2_NUM_RAYS), dtype=np.float64)
        ray_prey_mass = np.zeros((B, _V2_NUM_RAYS), dtype=np.float64)
        ray_threat    = np.zeros((B, _V2_NUM_RAYS), dtype=np.float64)
        ray_threat_mass = np.zeros((B, _V2_NUM_RAYS), dtype=np.float64)
        quad_threat   = np.zeros((B, 4), dtype=np.float64)
        mass_rank     = np.full(B, 0.5, dtype=np.float64)
        danger_score  = np.zeros(B, dtype=np.float64)

    # ---- Viruses: all-pairs reduction ----
    if virus_x.size:
        ray_virus = _allpairs_viruses(
            virus_x, virus_y, bot_cx, bot_cy, bot_scan, bot_alive, B,
        )
    else:
        ray_virus = np.zeros((B, _V2_NUM_RAYS), dtype=np.float64)

    # ---- Wall distances (batched analytical) ----
    ray_wall = _wall_rays_batch(bot_cx, bot_cy, bot_scan, B)

    # ---- Quadrant normalisation ----
    quad_food_n   = np.minimum(quad_food   * _V2_INV_QFOOD,   1.0)
    quad_threat_n = np.minimum(quad_threat * _V2_INV_QTHREAT, 1.0)

    # ---- Contextual ----
    cdx = bot_cx - config.WORLD_W * 0.5
    cdy = bot_cy - config.WORLD_H * 0.5
    center_dist = np.minimum(np.sqrt(cdx * cdx + cdy * cdy) * _V2_INV_HALF_DIAG, 1.0)

    # ---- Velocity ----
    vel_x = np.clip(last_vx * _V2_INV_VEL_NORM, -1.0, 1.0)
    vel_y = np.clip(last_vy * _V2_INV_VEL_NORM, -1.0, 1.0)

    # ---- Assemble (B, 122): layout mirrors _build_inputs_v2 ----
    # 5 intero + 8 rays × 7 (food,prey,prey_mass,threat,threat_mass,virus,wall) + 4 qfood + 4 qthreat + 3 ctx + 2 vel + MAX_CELLS*3 cell body
    out = np.empty((B, 122), dtype=np.float32)
    out[:, 0:5] = intero
    base = 5
    for r in range(_V2_NUM_RAYS):
        col = base + r * 7
        out[:, col + 0] = ray_food[:, r]
        out[:, col + 1] = ray_prey[:, r]
        out[:, col + 2] = ray_prey_mass[:, r]
        out[:, col + 3] = ray_threat[:, r]
        out[:, col + 4] = ray_threat_mass[:, r]
        out[:, col + 5] = ray_virus[:, r]
        out[:, col + 6] = ray_wall[:, r]
    out[:, 61:65] = quad_food_n
    out[:, 65:69] = quad_threat_n
    out[:, 69] = center_dist
    out[:, 70] = mass_rank
    out[:, 71] = danger_score
    out[:, 72] = vel_x
    out[:, 73] = vel_y

    # ---- Per-cell body features (74..121): sorted by mass desc, padded with zeros ----
    out[:, 74:] = 0.0
    for i, pid in enumerate(plan_pids):
        if not bot_alive[i]:
            continue
        player = world.players.get(pid)
        if player is None:
            continue
        cx_i = bot_cx[i]
        cy_i = bot_cy[i]
        inv_scan_i = 1.0 / max(float(bot_scan[i]), 1.0)
        sorted_c = sorted(player.cells, key=lambda c: c.mass, reverse=True)
        for ci, sc in enumerate(sorted_c[:_MAX_CELLS]):
            col = 74 + ci * 3
            out[i, col]     = float(np.clip((sc.x - cx_i) * inv_scan_i, -1.0, 1.0))
            out[i, col + 1] = float(np.clip((sc.y - cy_i) * inv_scan_i, -1.0, 1.0))
            out[i, col + 2] = float(math.log1p(sc.mass) / _V2_MAX_MASS_LOG)

    # Zero out dead bots so the network sees no stale activations
    if not bot_alive.all():
        out[~bot_alive] = 0.0
    return out


def _allpairs_food(food_x, food_y, bot_cx, bot_cy, bot_scan, bot_alive, B):
    """KDTree-based food perception. Returns (ray_food (B,8), quad_food (B,4)).

    Uses cKDTree.query_ball_point with per-bot scan radii so only the ~180
    in-range food items per bot are processed, vs the O(B×F) all-pairs matrix.
    """
    food_xy = np.column_stack([food_x, food_y])     # (F, 2)
    bot_pos = np.column_stack([bot_cx, bot_cy])      # (B, 2)
    results = cKDTree(food_xy).query_ball_point(bot_pos, r=bot_scan, p=2.0)  # (B,) of lists

    bot_list: list[int] = []
    seq_list: list[int] = []
    for b in range(B):
        if not bot_alive[b]:
            continue
        nbrs = results[b]
        if nbrs:
            n = len(nbrs)
            bot_list.extend([b] * n)
            seq_list.extend(nbrs)

    ray_food  = np.zeros((B, _V2_NUM_RAYS), dtype=np.float32)
    quad_food = np.zeros((B, 4),            dtype=np.float32)
    if not bot_list:
        return ray_food, quad_food

    b_arr = np.asarray(bot_list, dtype=np.int32)
    f_arr = np.asarray(seq_list, dtype=np.int32)
    dx    = food_x[f_arr] - bot_cx[b_arr]
    dy    = food_y[f_arr] - bot_cy[b_arr]
    dist  = np.sqrt(dx * dx + dy * dy)
    prox  = 1.0 - dist / bot_scan[b_arr]

    ang  = np.arctan2(dy, dx)
    ang[ang < 0.0] += _V2_TWO_PI
    sec  = (ang * _V2_INV_RAY_ANGLE).astype(np.int32) % _V2_NUM_RAYS

    right = (dx >= 0).astype(np.int32)
    down  = (dy >= 0).astype(np.int32)
    quads = 2 * down + (right == down).astype(np.int32)

    np.maximum.at(ray_food,  (b_arr, sec),   prox)
    np.add.at(quad_food,     (b_arr, quads), 1.0)
    return ray_food, quad_food


def _allpairs_cells(cell_x, cell_y, cell_m, cell_pid,
                    bot_cx, bot_cy, bot_scan, bot_pid_arr,
                    bot_largest, bot_mass, bot_alive, B):
    """KDTree-based cell perception — sparse edge list instead of O(B×N) matrix."""
    def _empty():
        z = np.zeros((B, _V2_NUM_RAYS), dtype=np.float64)
        return (z, z.copy(), z.copy(), z.copy(),
                np.zeros((B, 4), dtype=np.float64),
                np.full(B, 0.5, dtype=np.float64),
                np.zeros(B, dtype=np.float64))

    cell_xy = np.column_stack([cell_x, cell_y])
    bot_pos = np.column_stack([bot_cx, bot_cy])
    results = cKDTree(cell_xy).query_ball_point(bot_pos, r=bot_scan, p=2.0)

    bot_list: list[int] = []
    seq_list: list[int] = []
    for b in range(B):
        if not bot_alive[b]:
            continue
        nbrs = results[b]
        if nbrs:
            bot_list.extend([b] * len(nbrs))
            seq_list.extend(nbrs)

    if not bot_list:
        return _empty()

    b_arr = np.asarray(bot_list, dtype=np.int32)
    c_arr = np.asarray(seq_list, dtype=np.int32)

    # Remove same-player edges in numpy
    same = bot_pid_arr[b_arr] == cell_pid[c_arr]
    if same.all():
        return _empty()
    if same.any():
        b_arr = b_arr[~same]
        c_arr = c_arr[~same]

    dx   = cell_x[c_arr] - bot_cx[b_arr]
    dy   = cell_y[c_arr] - bot_cy[b_arr]
    dist = np.sqrt(dx * dx + dy * dy)
    prox = 1.0 - dist / bot_scan[b_arr]
    ang  = np.arctan2(dy, dx)
    ang  = np.where(ang < 0.0, ang + _V2_TWO_PI, ang)
    secs = (ang * _V2_INV_RAY_ANGLE).astype(np.int32) % _V2_NUM_RAYS

    cm_v      = cell_m[c_arr]
    largest_v = bot_largest[b_arr]
    own_v     = bot_mass[b_arr]
    eat_ratio = config.EAT_RATIO

    prey_mask   = (cm_v * eat_ratio) <= largest_v
    threat_mask = cm_v >= (largest_v * eat_ratio)

    visible_count_b = np.bincount(b_arr, minlength=B).astype(np.float64)
    prey_count_b    = (np.bincount(b_arr[prey_mask], minlength=B).astype(np.float64)
                       if prey_mask.any() else np.zeros(B, dtype=np.float64))

    ray_prey      = np.zeros((B, _V2_NUM_RAYS), dtype=np.float64)
    ray_prey_mass = np.zeros((B, _V2_NUM_RAYS), dtype=np.float64)
    if prey_mask.any():
        p_b    = b_arr[prey_mask]
        p_secs = secs[prey_mask]
        p_prox = prox[prey_mask]
        p_pm   = cm_v[prey_mask] / np.maximum(own_v[prey_mask], 1e-6)
        np.maximum.at(ray_prey, (p_b, p_secs), p_prox)
        flat_p = p_b * _V2_NUM_RAYS + p_secs
        order  = np.lexsort((p_prox, flat_p))
        ray_prey_mass.flat[flat_p[order]] = p_pm[order]

    ray_threat      = np.zeros((B, _V2_NUM_RAYS), dtype=np.float64)
    ray_threat_mass = np.zeros((B, _V2_NUM_RAYS), dtype=np.float64)
    quad_threat     = np.zeros((B, 4), dtype=np.float64)
    danger_b        = np.zeros(B, dtype=np.float64)
    if threat_mask.any():
        t_b    = b_arr[threat_mask]
        t_dx   = dx[threat_mask]
        t_dy   = dy[threat_mask]
        t_secs = secs[threat_mask]
        t_prox = prox[threat_mask]
        t_m    = cm_v[threat_mask]
        t_own  = own_v[threat_mask]
        t_tm   = np.minimum(t_own / np.maximum(t_m, 1e-6), 1.0)
        np.maximum.at(ray_threat, (t_b, t_secs), t_prox)
        flat_t = t_b * _V2_NUM_RAYS + t_secs
        order  = np.lexsort((t_prox, flat_t))
        ray_threat_mass.flat[flat_t[order]] = t_tm[order]
        right   = (t_dx >= 0).astype(np.int32)
        down    = (t_dy >= 0).astype(np.int32)
        t_quads = 2 * down + (right == down).astype(np.int32)
        np.add.at(quad_threat, (t_b, t_quads), t_m)
        np.add.at(danger_b, t_b, (t_m / np.maximum(t_own, 1e-6)) * (1.0 - t_prox))

    safe_visible = np.maximum(visible_count_b, 1.0)
    mass_rank_b  = np.where(visible_count_b > 0.0, prey_count_b / safe_visible, 0.5)
    danger_score = np.minimum(danger_b * _V2_INV_DANGER, 1.0)
    return ray_prey, ray_prey_mass, ray_threat, ray_threat_mass, quad_threat, mass_rank_b, danger_score


def _allpairs_viruses(virus_x, virus_y, bot_cx, bot_cy, bot_scan, bot_alive, B):
    """KDTree-based virus perception — sparse edge list instead of O(B×N) matrix."""
    virus_xy = np.column_stack([virus_x, virus_y])
    bot_pos  = np.column_stack([bot_cx, bot_cy])
    results  = cKDTree(virus_xy).query_ball_point(bot_pos, r=bot_scan, p=2.0)

    bot_list: list[int] = []
    seq_list: list[int] = []
    for b in range(B):
        if not bot_alive[b]:
            continue
        nbrs = results[b]
        if nbrs:
            bot_list.extend([b] * len(nbrs))
            seq_list.extend(nbrs)

    ray_virus = np.zeros((B, _V2_NUM_RAYS), dtype=np.float64)
    if not bot_list:
        return ray_virus

    b_arr = np.asarray(bot_list, dtype=np.int32)
    v_arr = np.asarray(seq_list, dtype=np.int32)
    dx    = virus_x[v_arr] - bot_cx[b_arr]
    dy    = virus_y[v_arr] - bot_cy[b_arr]
    dist  = np.sqrt(dx * dx + dy * dy)
    prox  = 1.0 - dist / bot_scan[b_arr]
    ang   = np.arctan2(dy, dx)
    ang   = np.where(ang < 0.0, ang + _V2_TWO_PI, ang)
    secs  = (ang * _V2_INV_RAY_ANGLE).astype(np.int32) % _V2_NUM_RAYS
    np.maximum.at(ray_virus, (b_arr, secs), prox)
    return ray_virus


def _wall_rays_batch(bot_cx, bot_cy, bot_scan, B):
    """Returns (B, NUM_RAYS) wall proximity in [0, 1]."""
    rdx = _V2_WALL_RDX_ARR  # (NUM_RAYS,)
    rdy = _V2_WALL_RDY_ARR
    cx = bot_cx[:, np.newaxis]   # (B, 1)
    cy = bot_cy[:, np.newaxis]
    sr = bot_scan[:, np.newaxis]
    big = 1e18
    # Distances to vertical and horizontal walls along each ray
    safe_rdx = np.where(rdx != 0, rdx, 1.0)
    safe_rdy = np.where(rdy != 0, rdy, 1.0)
    tx = np.where(rdx > 1e-9, (config.WORLD_W - cx) / safe_rdx,
         np.where(rdx < -1e-9, cx / -safe_rdx, big))
    ty = np.where(rdy > 1e-9, (config.WORLD_H - cy) / safe_rdy,
         np.where(rdy < -1e-9, cy / -safe_rdy, big))
    t = np.minimum(tx, ty)
    return np.maximum(0.0, 1.0 - t / sr)
