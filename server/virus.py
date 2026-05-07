from __future__ import annotations
from dataclasses import dataclass
import random
import math

from . import config
from .spatial import SpatialGrid


@dataclass
class Virus:
    id: int
    x: float
    y: float
    mass: float
    feed_count: int = 0  # how many times it's been fed


class VirusManager:
    def __init__(self, grid: SpatialGrid):
        self._grid = grid
        self._viruses: dict[int, Virus] = {}
        self._next_id: int = 1
        self._new_this_tick: list[int] = []
        self._removed_this_tick: list[int] = []

    def _alloc_id(self) -> int:
        vid = self._next_id
        self._next_id += 1
        return vid

    def count(self) -> int:
        return len(self._viruses)

    def get(self, virus_id: int) -> Virus | None:
        return self._viruses.get(virus_id)

    def spawn(self, x: float, y: float) -> int:
        """Spawn a single virus at the given position. Returns virus ID."""
        vid = self._alloc_id()
        virus = Virus(id=vid, x=x, y=y, mass=config.VIRUS_MASS)
        self._viruses[vid] = virus
        self._grid.insert(vid, x, y)
        self._new_this_tick.append(vid)
        return vid

    def spawn_corners(self) -> None:
        """Spawn viruses in all four corners of the world."""
        margin = config.VIRUS_CORNER_MARGIN
        corners = [
            (margin, margin),                                    # top-left
            (config.WORLD_W - margin, margin),                   # top-right
            (margin, config.WORLD_H - margin),                   # bottom-left
            (config.WORLD_W - margin, config.WORLD_H - margin),  # bottom-right
        ]
        for x, y in corners:
            self.spawn(x, y)

    def spawn_random(self) -> None:
        """Spawn a virus at a random location."""
        x = random.uniform(config.VIRUS_CORNER_MARGIN, config.WORLD_W - config.VIRUS_CORNER_MARGIN)
        y = random.uniform(config.VIRUS_CORNER_MARGIN, config.WORLD_H - config.VIRUS_CORNER_MARGIN)
        self.spawn(x, y)

    def feed(self, virus_id: int, mass_added: float) -> bool:
        """
        Feed a virus with ejected mass. Returns True if virus should shoot out a new one.
        When a virus is fed enough times, it ejects a new virus and resets.
        """
        virus = self._viruses.get(virus_id)
        if virus is None:
            return False
        
        virus.mass += mass_added
        virus.feed_count += 1
        
        # After enough feeds, the virus shoots out a new one
        return virus.feed_count >= config.VIRUS_FEED_THRESHOLD

    def shoot_virus(self, parent_id: int, direction_x: float, direction_y: float) -> int | None:
        """
        Eject a new virus from the parent in the given direction.
        Resets parent's feed count and mass.
        Returns new virus ID or None if parent doesn't exist.
        """
        parent = self._viruses.get(parent_id)
        if parent is None:
            return None
        
        # Reset parent
        parent.feed_count = 0
        parent.mass = config.VIRUS_MASS
        
        # Calculate spawn position outside parent
        mag = math.hypot(direction_x, direction_y) or 1.0
        nx = direction_x / mag
        ny = direction_y / mag
        
        spawn_dist = config.VIRUS_RADIUS * 2.5
        new_x = parent.x + nx * spawn_dist
        new_y = parent.y + ny * spawn_dist
        
        # Clamp to world bounds
        new_x = max(50, min(config.WORLD_W - 50, new_x))
        new_y = max(50, min(config.WORLD_H - 50, new_y))
        
        return self.spawn(new_x, new_y)

    def remove(self, virus_id: int) -> bool:
        """Remove a virus. Returns True if removed, False if not found."""
        virus = self._viruses.pop(virus_id, None)
        if virus is None:
            return False
        self._grid.remove(virus_id)
        self._removed_this_tick.append(virus_id)
        return True

    def flush_delta(self) -> tuple[list[Virus], list[int]]:
        """
        Returns (new_virus_list, removed_id_list) and clears the delta.
        Call once per tick after building all packets.
        """
        new_viruses = [self._viruses[vid] for vid in self._new_this_tick if vid in self._viruses]
        removed = self._removed_this_tick[:]
        self._new_this_tick.clear()
        self._removed_this_tick.clear()
        return new_viruses, removed

    def all_viruses(self) -> dict[int, Virus]:
        return self._viruses

    def respawn_to_target(self, target_count: int) -> None:
        """Spawn random viruses until we reach target count."""
        current = self.count()
        if current < target_count:
            for _ in range(target_count - current):
                self.spawn_random()
