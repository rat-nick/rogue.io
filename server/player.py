from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import websockets

from . import config


def cell_radius(mass: float) -> float:
    return math.sqrt(mass) * config.RADIUS_FACTOR


@dataclass
class Cell:
    id: int
    player_id: int
    x: float
    y: float
    mass: float
    vx: float = 0.0
    vy: float = 0.0
    merge_timer: float = 0.0         # seconds until this cell can merge (> 0 = not ready)
    collision_restore_ticks: int = 0 # ticks left where same-player cells skip push-apart
    # grid key cached to avoid recomputing (updated by SpatialGrid)
    grid_key: tuple = field(default=(0, 0), compare=False)

    def __setattr__(self, name: str, value) -> None:
        object.__setattr__(self, name, value)
        if name == 'mass':
            r = math.sqrt(max(value, 0.0))
            object.__setattr__(self, '_radius', r * config.RADIUS_FACTOR)
            size = max(1, math.ceil(10.0 * r))
            object.__setattr__(self, '_speed',
                max(config.MIN_SPEED, config.BASE_SPEED / (size ** config.SPEED_EXPONENT)))

    @property
    def radius(self) -> float:
        return self._radius

    @property
    def speed(self) -> float:
        return self._speed


@dataclass
class Player:
    id: int
    name: str
    websocket: object   # websockets.ServerConnection — avoid import cycle
    cells: list[Cell] = field(default_factory=list)
    target_x: float = 0.0
    target_y: float = 0.0
    split_pending: bool = False
    eject_pending: bool = False
    hue: int = -1       # genetic hue (0-359); -1 = derive from player_id on client
    # name caching: player IDs whose names have been sent to this client
    known_player_ids: set[int] = field(default_factory=set)
    # delta tracking for food and viruses
    sent_food_ids: set[int] = field(default_factory=set)
    sent_virus_ids: set[int] = field(default_factory=set)


    @property
    def total_mass(self) -> float:
        return sum(c.mass for c in self.cells)

    @property
    def score(self) -> int:
        return int(self.total_mass)

    @property
    def centroid(self) -> tuple[float, float]:
        if not self.cells:
            return (config.WORLD_W / 2, config.WORLD_H / 2)
        total = 0.0
        wx = 0.0
        wy = 0.0
        for c in self.cells:
            m = c.mass
            total += m
            wx += c.x * m
            wy += c.y * m
        if total == 0.0:
            return (config.WORLD_W / 2, config.WORLD_H / 2)
        return (wx / total, wy / total)

    @property
    def viewport_rect(self) -> tuple[float, float, float, float]:
        """Returns (x, y, w, h) in world coordinates."""
        cx, cy = self.centroid
        total = max(self.total_mass, 1.0)
        # scale viewport with mass so larger cells see more of the world
        vw = config.VIEW_BASE_SIZE * ((total / 100.0) ** config.VIEW_MASS_SCALE)
        # assume 16:9 aspect
        vh = vw * 9.0 / 16.0
        # add padding for large cells near viewport edges
        from . import config as cfg
        max_r = max((c.radius for c in self.cells), default=0) + cfg.RADIUS_FACTOR * 20
        vw += max_r * 2
        vh += max_r * 2
        return (cx - vw / 2, cy - vh / 2, vw, vh)
