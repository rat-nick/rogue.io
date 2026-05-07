from __future__ import annotations
from dataclasses import dataclass
import random

from . import config
from .spatial import SpatialGrid


@dataclass
class Food:
    id: int
    x: float
    y: float
    color_idx: int
    mass: float
    is_remnant: bool = False
    vx: float = 0.0  # velocity for ejected mass
    vy: float = 0.0


class FoodManager:
    def __init__(self, grid: SpatialGrid):
        self._grid = grid
        self._food: dict[int, Food] = {}
        self._next_id: int = 1
        self._new_this_tick: list[int] = []
        self._removed_this_tick: list[int] = []

    # ------------------------------------------------------------------

    def _alloc_id(self) -> int:
        fid = self._next_id
        self._next_id += 1
        return fid

    def count(self) -> int:
        return len(self._food)

    def get(self, food_id: int) -> Food | None:
        return self._food.get(food_id)

    def spawn_batch(self, n: int) -> None:
        """Spawn n food pellets at random world positions."""
        for _ in range(n):
            x = random.uniform(0, config.WORLD_W)
            y = random.uniform(0, config.WORLD_H)
            color_idx = random.randrange(len(config.FOOD_COLORS))
            fid = self._alloc_id()
            food = Food(id=fid, x=x, y=y, color_idx=color_idx, mass=config.FOOD_MASS)
            self._food[fid] = food
            self._grid.insert(fid, x, y)
            self._new_this_tick.append(fid)

    def spawn_ejected(self, x: float, y: float, color_idx: int, mass: float, vx: float = 0.0, vy: float = 0.0) -> int:
        """Spawn a single food pellet (used for ejected mass). Returns food ID."""
        fid = self._alloc_id()
        food = Food(id=fid, x=x, y=y, color_idx=color_idx, mass=mass, vx=vx, vy=vy)
        self._food[fid] = food
        self._grid.insert(fid, x, y)
        self._new_this_tick.append(fid)
        return fid

    def spawn_remnant(self, x: float, y: float, color_idx: int, mass: float) -> None:
        fid = self._alloc_id()
        food = Food(id=fid, x=x, y=y, color_idx=color_idx, mass=mass, is_remnant=True)
        self._food[fid] = food
        self._grid.insert(fid, x, y)
        self._new_this_tick.append(fid)

    def tick_decay(self, dt: float) -> None:
        """Update food decay and ejected mass physics."""
        to_remove: list[int] = []
        for fid, food in self._food.items():
            # Decay remnants
            if food.is_remnant:
                food.mass *= (1.0 - config.REMNANT_DECAY_RATE * dt)
                if food.mass < config.REMNANT_MIN_MASS:
                    to_remove.append(fid)
            
            # Update ejected mass positions and decelerate
            if food.vx != 0.0 or food.vy != 0.0:
                food.x += food.vx * dt
                food.y += food.vy * dt
                # Clamp to world bounds
                food.x = max(0.0, min(config.WORLD_W, food.x))
                food.y = max(0.0, min(config.WORLD_H, food.y))
                # Decelerate
                decel_factor = max(0.0, 1.0 - config.EJECT_DECEL * dt)
                food.vx *= decel_factor
                food.vy *= decel_factor
                # Stop when very slow
                if abs(food.vx) < 1.0 and abs(food.vy) < 1.0:
                    food.vx = 0.0
                    food.vy = 0.0
                # Update grid position
                self._grid.move(fid, food.x, food.y)
        
        for fid in to_remove:
            self.eat(fid)

    def eat(self, food_id: int) -> float:
        """Remove food and return its mass. Returns 0 if not found."""
        food = self._food.pop(food_id, None)
        if food is None:
            return 0.0
        self._grid.remove(food_id)
        self._removed_this_tick.append(food_id)
        return food.mass

    def flush_delta(self) -> tuple[list[Food], list[int]]:
        """
        Returns (new_food_list, removed_id_list) and clears the delta.
        Call once per tick after building all packets.
        """
        new_food = [self._food[fid] for fid in self._new_this_tick if fid in self._food]
        removed = self._removed_this_tick[:]
        self._new_this_tick.clear()
        self._removed_this_tick.clear()
        return new_food, removed

    def all_food(self) -> dict[int, Food]:
        return self._food
