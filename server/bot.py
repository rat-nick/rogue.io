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

_SCAN_RADIUS   = 3000.0           # world units — how far the bot can sense
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

    def register(self, player_id: int, start_x: float, start_y: float,
                 genome: neat.DefaultGenome) -> None:
        self._state[player_id] = {
            'genome':           genome,
            'start_mass':       20.0,
            'ticks_alive':      0,
            'peak_mass':        20.0,
            'last_split_tick':  0,
            'deaths':           0,
            'last_pos':         (start_x, start_y),
            'distance_traveled': 0.0,
            'last_mass':        20.0,
            'food_eaten_count': 0,
            'idle_ticks':       0,
            'corner_ticks':     0,
        }
        self._plan = None

    def unregister(self, player_id: int) -> tuple[neat.DefaultGenome | None, float]:
        """Remove bot state and return (genome, fitness). Returns (None, 0) if unknown."""
        bs = self._state.pop(player_id, None)
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
        mass_gained_per_sec = (bs['peak_mass'] - bs['start_mass']) / time_alive
        
        # Base fitness: peak mass + mass gain rate - death penalty
        fitness = bs['peak_mass'] + 20.0 * mass_gained_per_sec - 10000.0 * bs['deaths']
        
        # Reward active food seeking
        food_bonus = bs['food_eaten_count'] * 5.0
        
        # Reward movement (distance traveled)
        distance_bonus = bs['distance_traveled'] * 0.01
        
        # Penalize idleness (staying still)
        ticks_alive = max(bs['ticks_alive'], 1)
        idle_ratio = bs['idle_ticks'] / ticks_alive
        idle_penalty = idle_ratio * 200.0
        
        # Penalize corner camping
        corner_ratio = bs['corner_ticks'] / ticks_alive
        corner_penalty = corner_ratio * 300.0
        
        return fitness + food_bonus + distance_bonus - idle_penalty - corner_penalty

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
            self._plan = build_batch_plan(pids, genomes, neat_config)

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
            
            # Track movement and idleness
            cx, cy = player.centroid
            last_x, last_y = bs['last_pos']
            dist_moved = math.hypot(cx - last_x, cy - last_y)
            bs['distance_traveled'] += dist_moved
            bs['last_pos'] = (cx, cy)
            
            # Track idleness (moving less than 5 units per tick)
            if dist_moved < 5.0:
                bs['idle_ticks'] += 1
            
            # Track corner camping (within 200 units of any corner)
            corner_margin = 200.0
            near_corner = (
                (cx < corner_margin or cx > config.WORLD_W - corner_margin) and
                (cy < corner_margin or cy > config.WORLD_H - corner_margin)
            )
            if near_corner:
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
            player.target_x = max(100.0, min(config.WORLD_W - 100.0,
                                             cx + move_x * _TARGET_DIST))
            player.target_y = max(100.0, min(config.WORLD_H - 100.0,
                                             cy + move_y * _TARGET_DIST))
            split_pending = float(outputs_arr[i, 2]) > 0.0
            if split_pending:
                bs['last_split_tick'] = bs['ticks_alive']
            player.split_pending = split_pending
            player.eject_pending = float(outputs_arr[i, 3]) > 0.0


# ---------------------------------------------------------------------------
# Sensory encoding — 8 sectors × 6 values = 48 inputs
# ---------------------------------------------------------------------------
#
# Sector layout (directions from the bot's position):
#   Sector 0: East  (0°–45°),  Sector 1: NE (45°–90°), … (counter-clockwise)
#
# Per sector (indices 0–7):
#   [0] food_proximity   — 1 − dist/SCAN_RADIUS  (0 = nothing nearby)
#   [1] food_mass_norm   — food.mass / _FOOD_MASS_NORM, clamped 0–1
#   [2] prey_proximity   — 1 − dist/SCAN_RADIUS  (nearest smaller cell)
#   [3] prey_smallness   — 1 − prey_mass/own_mass (0 = same size, 1 = tiny)
#   [4] threat_proximity — 1 − dist/SCAN_RADIUS  (nearest larger cell)
#   [5] threat_danger    — 1 − own_mass/threat_mass (0 = just bigger, 1 = enormous)
#   [6] virus_proximity  — 1 − dist/SCAN_RADIUS  (nearest virus)
#   [7] virus_danger     — 1.0 if any cell > 150 mass (will split), else 0.0

def _build_inputs(world: 'GameWorld', player, bot_state: dict) -> list[float]:
    if not player.cells:
        return [0.0] * (_NUM_SECTORS * 8 + _MAX_CELLS * 3 + 1)

    cx, cy = player.centroid
    own_mass = player.total_mass
    largest_cell_mass = max(c.mass for c in player.cells)

    sectors          = [[0.0] * 8 for _ in range(_NUM_SECTORS)]
    food_best_dist   = [_SCAN_RADIUS] * _NUM_SECTORS
    prey_best_dist   = [_SCAN_RADIUS] * _NUM_SECTORS
    threat_best_dist = [_SCAN_RADIUS] * _NUM_SECTORS
    virus_best_dist  = [_SCAN_RADIUS] * _NUM_SECTORS

    # --- Food ---
    for fid in world.food_grid.query_radius(cx, cy, _SCAN_RADIUS):
        food = world.food_mgr.get(fid)
        if food is None:
            continue
        dx, dy = food.x - cx, food.y - cy
        dist = math.hypot(dx, dy)
        if dist < 1e-6 or dist >= _SCAN_RADIUS:
            continue
        s = _sector(dx, dy)
        if dist < food_best_dist[s]:
            food_best_dist[s] = dist
            sectors[s][0] = 1.0 - dist / _SCAN_RADIUS
            sectors[s][1] = min(food.mass / _FOOD_MASS_NORM, 1.0)

    # --- Other cells ---
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
        s = _sector(dx, dy)
        other_mass = other_player.total_mass
        if other_mass < own_mass:  # prey
            if dist < prey_best_dist[s]:
                prey_best_dist[s] = dist
                sectors[s][2] = 1.0 - dist / _SCAN_RADIUS
                sectors[s][3] = 1.0 - other_mass / own_mass
        else:  # threat
            if dist < threat_best_dist[s]:
                threat_best_dist[s] = dist
                sectors[s][4] = 1.0 - dist / _SCAN_RADIUS
                sectors[s][5] = 1.0 - min(own_mass / other_mass, 1.0)

    # --- Viruses ---
    # Check if any cell can be split by a virus (mass > config.VIRUS_SPLIT_THRESHOLD)
    has_vulnerable_cell = any(c.mass > config.VIRUS_SPLIT_THRESHOLD for c in player.cells)
    virus_danger_val = 1.0 if has_vulnerable_cell else 0.0

    if hasattr(world, 'virus_grid') and world.virus_grid is not None:
        for vid in world.virus_grid.query_radius(cx, cy, _SCAN_RADIUS):
            virus = world.virus_mgr.get(vid)
            if virus is None:
                continue
            dx, dy = virus.x - cx, virus.y - cy
            dist = math.hypot(dx, dy)
            if dist < 1e-6 or dist >= _SCAN_RADIUS:
                continue
            s = _sector(dx, dy)
            if dist < virus_best_dist[s]:
                virus_best_dist[s] = dist
                sectors[s][6] = 1.0 - dist / _SCAN_RADIUS
                sectors[s][7] = virus_danger_val

    inputs = [v for s in sectors for v in s]

    sorted_cells = sorted(player.cells, key=lambda c: c.mass, reverse=True)
    own_mass_safe = max(own_mass, 1e-6)

    for i in range(_MAX_CELLS):
        if i < len(sorted_cells):
            inputs.append(sorted_cells[i].mass / own_mass_safe)
        else:
            inputs.append(0.0)

    for i in range(_MAX_CELLS):
        if i < len(sorted_cells):
            c = sorted_cells[i]
            rel_x = max(-1.0, min(1.0, (c.x - cx) / _CELL_POS_NORM))
            rel_y = max(-1.0, min(1.0, (c.y - cy) / _CELL_POS_NORM))
            inputs.append(rel_x)
            inputs.append(rel_y)
        else:
            inputs.append(0.0)
            inputs.append(0.0)

    ticks_alive = bot_state['ticks_alive']
    last_split_tick = bot_state['last_split_tick']
    time_since_split = max(ticks_alive - last_split_tick, 0) * config.TICK_INTERVAL
    inputs.append(min(time_since_split / _SPLIT_TIME_NORM, 1.0))

    return inputs


def _sector(dx: float, dy: float) -> int:
    angle = math.atan2(dy, dx) % (2.0 * math.pi)
    return int(angle / _SECTOR_ANGLE) % _NUM_SECTORS
