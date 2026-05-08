from __future__ import annotations
import math
from typing import TYPE_CHECKING

import neat
import numpy as np

from . import config
from .genetics import neat_config
from .nn_batch import build_batch_plan, BatchPlan

if TYPE_CHECKING:
    from .game import GameWorld


# ---------------------------------------------------------------------------
# Sensory constants
# ---------------------------------------------------------------------------

_SCAN_RADIUS   = 1500.0           # world units — how far the bot can sense
_NUM_SECTORS   = 16               # directional bins around the bot
_SECTOR_ANGLE  = 2.0 * math.pi / _NUM_SECTORS
_TARGET_DIST   = 2000.0           # world units to project the move vector
_FOOD_MASS_NORM = 50.0            # food mass normalisation factor
_MAX_CELLS = config.MAX_CELLS
_CELL_POS_NORM = 1500.0
_SPLIT_TIME_NORM = 30.0
_VIRUS_RADIUS = config.VIRUS_RADIUS


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

    def register(self, player_id: int, start_x: float, start_y: float,
                 genome: neat.DefaultGenome) -> None:
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
            'idle_ticks':              0,
            'edge_ticks':              0,
            'corner_ticks':            0,
        }
        self._prev_plan = self._plan
        self._plan = None

    def unregister(self, player_id: int) -> tuple[neat.DefaultGenome | None, float]:
        """Remove bot state and return (genome, fitness). Returns (None, 0) if unknown."""
        bs = self._state.pop(player_id, None)
        self._prev_plan = self._plan
        self._plan = None
        if bs is None:
            return None, 0.0
        return bs['genome'], self._compute_fitness(bs)

    def current_fitness(self, player_id: int) -> float:
        """Return current fitness without removing bot state. Returns 0 if unknown."""
        bs = self._state.get(player_id)
        if bs is None:
            return 0.0
        return self._compute_fitness(bs)

    def _compute_fitness(self, bs: dict) -> float:
        time_alive = max(bs['ticks_alive'] * config.TICK_INTERVAL, 1e-6)
        ticks_alive = max(bs['ticks_alive'], 1)

        peak_mass  = bs['peak_mass']
        avg_mass   = bs['total_mass_accumulated'] / ticks_alive
        total_food = bs['food_eaten_count']

        edge_fraction   = bs['edge_ticks']   / ticks_alive
        corner_fraction = bs['corner_ticks'] / ticks_alive
        corner_penalty  = (edge_fraction ** 2 + corner_fraction ** 2) * 150.0

        survival_bonus = math.log1p(time_alive)
        fitness = (
            peak_mass  * 0.6
            + avg_mass * 0.4
            + total_food * 2.0
            + survival_bonus * 5.0
            - corner_penalty
        )

        if bs.get('deaths', 0) > 0:
            fitness *= 0.3

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
            self._plan = build_batch_plan(pids, genomes, neat_config, prior_plan=self._prev_plan)
            self._prev_plan = None

        # Collect inputs and update fitness tracking
        inputs_list: list[list[float]] = []
        valid_pids:  list[int]         = []
        for pid in self._plan.player_ids:
            bs     = self._state.get(pid)
            player = world.players.get(pid)
            if bs is None or player is None:
                self._plan = None  # stale — rebuild next tick
                return
            bs['ticks_alive'] += 1
            total_mass = player.total_mass
            if total_mass > bs['peak_mass']:
                bs['peak_mass'] = total_mass
            bs['total_mass_accumulated'] += total_mass
            
            # Track movement and idleness
            cx, cy = player.centroid
            last_x, last_y = bs['last_pos']
            dist_moved = math.hypot(cx - last_x, cy - last_y)
            bs['distance_traveled'] += dist_moved
            bs['last_pos'] = (cx, cy)
            
            # Track idleness: threshold is below MIN_SPEED/TICK_RATE so large slow cells
            # moving at their physics-limited speed are not penalised.
            if dist_moved < 1.5:
                bs['idle_ticks'] += 1
            
            # Track edge camping (within 1500 units of any wall)
            _cw, _ch = config.WORLD_W, config.WORLD_H
            if cx < 1500.0 or cx > _cw - 1500.0 or cy < 1500.0 or cy > _ch - 1500.0:
                bs['edge_ticks'] += 1

            # Track corner camping (within 2000 units of any corner); fires on top of edge
            if min(math.hypot(cx, cy), math.hypot(_cw - cx, cy),
                   math.hypot(cx, _ch - cy), math.hypot(_cw - cx, _ch - cy)) < 2000.0:
                bs['corner_ticks'] += 1
            
            # Estimate food eaten (mass increase of small amounts, likely food)
            mass_delta = total_mass - bs['last_mass']
            if 0 < mass_delta < 15.0:  # Food is typically small mass gains
                bs['food_eaten_count'] += 1
            bs['last_mass'] = total_mass
            
            inputs_list.append(_build_inputs(world, player, bs))
            valid_pids.append(pid)

        inputs_batch = np.array(inputs_list, dtype=np.float32)
        outputs_arr  = self._plan.run(inputs_batch)

        # Apply outputs
        for i, pid in enumerate(valid_pids):
            player = world.players[pid]
            bs = self._state[pid]
            move_x = float(outputs_arr[i, 0])
            move_y = float(outputs_arr[i, 1])
            mag = math.hypot(move_x, move_y)
            if mag > 1e-6:
                move_x /= mag
                move_y /= mag
            cx, cy = player.centroid
            player.target_x = max(0.0, min(float(config.WORLD_W), cx + move_x * _TARGET_DIST))
            player.target_y = max(0.0, min(float(config.WORLD_H), cy + move_y * _TARGET_DIST))
            split_pending = float(outputs_arr[i, 2]) > 0.0
            if split_pending:
                bs['last_split_tick'] = bs['ticks_alive']
            player.split_pending = split_pending
            player.eject_pending = float(outputs_arr[i, 3]) > 0.0


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def _build_inputs(world: 'GameWorld', player, bot_state: dict) -> list[float]:
    if config.PERCEPTION_VERSION == 2:
        return _build_inputs_v2(world, player, bot_state)
    return _build_inputs_v1(world, player, bot_state)


# ---------------------------------------------------------------------------
# Perception v1 — 16-sector spatial encoding (177 inputs)
# ---------------------------------------------------------------------------
#
# Layout: 16 sectors × 8 values = 128, + 16 cell masses, + 16 × 2 cell
# positions = 32, + 1 split timer  →  177 inputs total.
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

def _build_inputs_v1(world: 'GameWorld', player, bot_state: dict) -> list[float]:
    if not player.cells:
        return [0.0] * (_NUM_SECTORS * 8 + _MAX_CELLS * 3 + 1)

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

    return inputs


def _sector_v1(dx: float, dy: float) -> int:
    angle = math.atan2(dy, dx) % (2.0 * math.pi)
    return int(angle / _SECTOR_ANGLE) % _NUM_SECTORS


# ---------------------------------------------------------------------------
# Perception v2 — raycast + quadrant model (56 inputs)
# ---------------------------------------------------------------------------
#
# Layout:
#   5   interoceptive   (avg_mass_per_cell, num_cells, split/eject ready, merge timer)
#   40  raycasts        (8 rays × 5: food, prey, threat, virus, wall proximity)
#   8   quadrant sums   (4 quadrants × food_density + threat_level)
#   3   contextual      (center dist, mass rank, danger score)
#   ─── total: 56
#
# Ray encoding: 0.0 = nothing in range / far, 1.0 = object at bot centroid.
# Higher activation always means "pay attention here."

_V2_NUM_RAYS        = 8
_V2_RAY_ANGLE       = 2.0 * math.pi / _V2_NUM_RAYS
_V2_MAX_MASS_LOG    = math.log1p(5000.0)          # log-scale ceiling for mass
_V2_MERGE_TIMER_MAX = config.MERGE_TIME_BASE + config.MERGE_TIME_MASS_FACTOR * 5000.0
_V2_QUAD_FOOD_NORM  = 20.0    # expected max food pellets per quadrant in scan area
_V2_QUAD_THREAT_NORM = 5000.0 # expected max threat mass per quadrant
_V2_DANGER_NORM     = 10.0    # normalization divisor for danger_score


def _build_inputs_v2(world: 'GameWorld', player, _bot_state: dict) -> list[float]:
    if not player.cells:
        return [0.0] * 56

    cx, cy = player.centroid
    own_mass = player.total_mass
    largest_cell_mass = max(c.mass for c in player.cells)

    # --- Interoceptive (5) ---
    avg_mass       = own_mass / len(player.cells)
    avg_mass_n     = math.log1p(avg_mass) / _V2_MAX_MASS_LOG
    num_cells_n    = len(player.cells) / _MAX_CELLS
    split_ready    = 1.0 if any(c.mass >= config.MIN_SPLIT_MASS for c in player.cells) else 0.0
    eject_ready    = 1.0 if any(c.mass >= config.EJECT_MIN_MASS for c in player.cells) else 0.0
    max_merge      = max((c.merge_timer for c in player.cells), default=0.0)
    merge_timer_n  = min(max_merge / _V2_MERGE_TIMER_MAX, 1.0)

    # --- Raycast & quadrant accumulators ---
    ray_food   = [0.0] * _V2_NUM_RAYS   # 0.0 = nothing in range
    ray_prey   = [0.0] * _V2_NUM_RAYS
    ray_threat = [0.0] * _V2_NUM_RAYS
    ray_virus  = [0.0] * _V2_NUM_RAYS
    quad_food_count  = [0.0] * 4
    quad_threat_mass = [0.0] * 4
    visible_masses: list[float] = []
    danger_sum = 0.0

    # --- Food ---
    for fid in world.food_grid.query_radius(cx, cy, _SCAN_RADIUS):
        food = world.food_mgr.get(fid)
        if food is None:
            continue
        dx, dy = food.x - cx, food.y - cy
        dist = math.hypot(dx, dy)
        if dist < 1e-6 or dist >= _SCAN_RADIUS:
            continue
        dn = 1.0 - dist / _SCAN_RADIUS
        s = _sector_v2(dx, dy)
        if dn > ray_food[s]:
            ray_food[s] = dn
        quad_food_count[_quadrant(dx, dy)] += 1.0

    # --- Enemy cells ---
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
        ocm = other_cell.mass
        dn  = 1.0 - dist / _SCAN_RADIUS
        s   = _sector_v2(dx, dy)
        visible_masses.append(ocm)
        if ocm * config.EAT_RATIO <= largest_cell_mass:
            if dn > ray_prey[s]:
                ray_prey[s] = dn
        elif ocm >= largest_cell_mass * config.EAT_RATIO:
            if dn > ray_threat[s]:
                ray_threat[s] = dn
            quad_threat_mass[_quadrant(dx, dy)] += ocm
            danger_sum += (ocm / max(own_mass, 1e-6)) * (1.0 - dn)

    # --- Viruses ---
    if world.virus_grid is not None:
        for vid in world.virus_grid.query_radius(cx, cy, _SCAN_RADIUS):
            virus = world.virus_mgr.get(vid)
            if virus is None:
                continue
            dx, dy = virus.x - cx, virus.y - cy
            dist = math.hypot(dx, dy)
            if dist < 1e-6 or dist >= _SCAN_RADIUS:
                continue
            dn = 1.0 - dist / _SCAN_RADIUS
            s  = _sector_v2(dx, dy)
            if dn > ray_virus[s]:
                ray_virus[s] = dn

    # --- Wall distances (analytical) ---
    ray_wall = []
    for i in range(_V2_NUM_RAYS):
        angle = i * _V2_RAY_ANGLE
        rdx, rdy = math.cos(angle), math.sin(angle)
        t = float('inf')
        if rdx > 1e-9:
            t = min(t, (config.WORLD_W - cx) / rdx)
        elif rdx < -1e-9:
            t = min(t, cx / (-rdx))
        if rdy > 1e-9:
            t = min(t, (config.WORLD_H - cy) / rdy)
        elif rdy < -1e-9:
            t = min(t, cy / (-rdy))
        ray_wall.append(max(0.0, 1.0 - t / _SCAN_RADIUS))

    # --- Assemble raycasts (40) ---
    raycasts: list[float] = []
    for i in range(_V2_NUM_RAYS):
        raycasts += [ray_food[i], ray_prey[i], ray_threat[i], ray_virus[i], ray_wall[i]]

    # --- Quadrant summaries (8) ---
    quad_inputs = (
        [min(quad_food_count[q]  / _V2_QUAD_FOOD_NORM,  1.0) for q in range(4)] +
        [min(quad_threat_mass[q] / _V2_QUAD_THREAT_NORM, 1.0) for q in range(4)]
    )

    # --- Contextual (3) ---
    half_diag    = math.hypot(config.WORLD_W / 2.0, config.WORLD_H / 2.0)
    center_dist  = min(math.hypot(cx - config.WORLD_W / 2.0, cy - config.WORLD_H / 2.0) / half_diag, 1.0)
    mass_rank    = (sum(1 for m in visible_masses if m * config.EAT_RATIO <= largest_cell_mass)
                    / len(visible_masses)) if visible_masses else 0.5
    danger_score = min(danger_sum / _V2_DANGER_NORM, 1.0)

    return (
        [avg_mass_n, num_cells_n, split_ready, eject_ready, merge_timer_n]
        + raycasts
        + quad_inputs
        + [center_dist, mass_rank, danger_score]
    )


def _sector_v2(dx: float, dy: float) -> int:
    angle = math.atan2(dy, dx) % (2.0 * math.pi)
    return int(angle / _V2_RAY_ANGLE) % _V2_NUM_RAYS


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
