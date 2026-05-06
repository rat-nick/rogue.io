from __future__ import annotations
import math
import random
from typing import TYPE_CHECKING

from . import config
from .genetics import BotGenome, random_genome

if TYPE_CHECKING:
    from .game import GameWorld


class NullWebSocket:
    """Fake websocket for bots — silently discards all sends."""
    async def send(self, data) -> None:
        pass


class BotController:
    """
    Genome-driven bot AI.
    Each bot has a BotGenome that parameterises its decision thresholds.
    Fitness is tracked per-bot and reported on unregister (death/removal).
    """

    def __init__(self) -> None:
        # player_id -> bot state dict
        self._state: dict[int, dict] = {}

    def register(self, player_id: int, start_x: float, start_y: float,
                 genome: BotGenome) -> None:
        self._state[player_id] = {
            'genome':       genome,
            'wx':           start_x,
            'wy':           start_y,
            'wander_timer': random.uniform(0.0, genome.wander_interval),
            'split_timer':  random.uniform(genome.split_cooldown * 0.5,
                                           genome.split_cooldown),
            'ticks_alive':  0,
            'peak_mass':    20.0,
        }

    def unregister(self, player_id: int) -> tuple[BotGenome | None, float]:
        """Remove bot state and return (genome, fitness). Returns (None, 0) if unknown."""
        bs = self._state.pop(player_id, None)
        if bs is None:
            return None, 0.0
        fitness = self._compute_fitness(bs)
        return bs['genome'], fitness

    def get_genome(self, player_id: int) -> BotGenome | None:
        bs = self._state.get(player_id)
        return bs['genome'] if bs else None

    # ------------------------------------------------------------------

    def update(self, world: 'GameWorld', dt: float) -> None:
        for player_id, bs in list(self._state.items()):
            player = world.players.get(player_id)
            if player is None:
                continue

            bs['ticks_alive'] += 1
            total_mass = player.total_mass
            if total_mass > bs['peak_mass']:
                bs['peak_mass'] = total_mass

            genome: BotGenome = bs['genome']
            bs['wander_timer'] -= dt
            bs['split_timer']  -= dt

            if bs['wander_timer'] <= 0:
                self._pick_target(world, player, bs, genome)

            player.target_x = bs['wx']
            player.target_y = bs['wy']

            if (bs['split_timer'] <= 0
                    and player.total_mass >= genome.split_mass_threshold):
                player.split_pending = True
                bs['split_timer'] = genome.split_cooldown

    # ------------------------------------------------------------------

    @staticmethod
    def _compute_fitness(bs: dict) -> float:
        # Reward: peak mass achieved (main driver) + survival time (tiebreaker)
        return bs['peak_mass'] * 2.0 + bs['ticks_alive'] * 0.01

    @staticmethod
    def _pick_target(world: 'GameWorld', player, bs: dict,
                     genome: BotGenome) -> None:
        if not player.cells:
            return

        cx, cy = player.centroid
        own_mass = player.total_mass

        # 1. Chase chaseable prey
        nearby_cell_ids = world.cell_grid.query_radius(cx, cy, genome.prey_chase_radius)
        best_prey = None
        best_prey_dist = float('inf')
        for cid in nearby_cell_ids:
            entry = world.cell_map.get(cid)
            if entry is None:
                continue
            other_cell, other_player = entry
            if other_player.id == player.id:
                continue
            if other_player.total_mass < own_mass * genome.chase_mass_ratio:
                dist = math.hypot(cx - other_cell.x, cy - other_cell.y)
                if dist < best_prey_dist:
                    best_prey_dist = dist
                    best_prey = other_cell

        if best_prey is not None:
            bs['wx'] = best_prey.x
            bs['wy'] = best_prey.y
            bs['wander_timer'] = random.uniform(
                0.5, max(0.5, genome.wander_interval * 0.5))
            return

        # 2. Flee threats
        for cid in nearby_cell_ids:
            entry = world.cell_map.get(cid)
            if entry is None:
                continue
            other_cell, other_player = entry
            if other_player.id == player.id:
                continue
            if other_player.total_mass > own_mass * genome.flee_mass_ratio:
                dist = math.hypot(cx - other_cell.x, cy - other_cell.y)
                if dist < genome.threat_flee_radius:
                    dx = cx - other_cell.x
                    dy = cy - other_cell.y
                    d = math.hypot(dx, dy) or 1.0
                    bs['wx'] = max(100.0, min(config.WORLD_W - 100.0,
                                             cx + dx / d * 2000.0))
                    bs['wy'] = max(100.0, min(config.WORLD_H - 100.0,
                                             cy + dy / d * 2000.0))
                    bs['wander_timer'] = random.uniform(
                        0.5, genome.wander_interval)
                    return

        # 3. Seek highest-mass edible nearby food
        nearby_food_ids = world.food_grid.query_radius(cx, cy,
                                                       genome.food_seek_radius)
        if nearby_food_ids:
            largest_cell_mass = max(c.mass for c in player.cells)
            best_food = None
            best_food_mass = float('-inf')

            for fid in nearby_food_ids:
                food = world.food_mgr.get(fid)
                if food is None:
                    continue
                if food.is_remnant and food.mass >= largest_cell_mass:
                    continue
                if food.mass > best_food_mass:
                    best_food_mass = food.mass
                    best_food = food

            if best_food is not None:
                bs['wx'] = best_food.x
                bs['wy'] = best_food.y
                bs['wander_timer'] = random.uniform(
                    0.5, genome.wander_interval)
                return

        # 4. Random wander
        angle = random.uniform(0, 2 * math.pi)
        dist  = random.uniform(400, 3000)
        bs['wx'] = max(100.0, min(config.WORLD_W - 100.0,
                                  cx + math.cos(angle) * dist))
        bs['wy'] = max(100.0, min(config.WORLD_H - 100.0,
                                  cy + math.sin(angle) * dist))
        bs['wander_timer'] = genome.wander_interval
