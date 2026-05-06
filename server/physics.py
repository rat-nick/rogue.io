from __future__ import annotations
import math
import random
from typing import TYPE_CHECKING

from . import config
from .player import Cell, Player, cell_radius
from .spatial import SpatialGrid

if TYPE_CHECKING:
    from .food import FoodManager


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def apply_input(player: Player, dt: float) -> None:
    """Compute velocity for each cell toward player's target and move it."""
    for cell in player.cells:
        dx = player.target_x - cell.x
        dy = player.target_y - cell.y
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            cell.vx = 0.0
            cell.vy = 0.0
            continue
        nx = dx / dist
        ny = dy / dist
        spd = cell.speed
        # Don't overshoot: cap displacement to remaining distance
        move = min(spd * dt, dist)
        cell.x = max(0.0, min(config.WORLD_W, cell.x + nx * move))
        cell.y = max(0.0, min(config.WORLD_H, cell.y + ny * move))
        cell.vx = nx * spd
        cell.vy = ny * spd


def update_positions(cells: list[Cell], cell_grid: SpatialGrid) -> None:
    """Sync spatial grid after all cells have moved."""
    for cell in cells:
        cell_grid.move(cell.id, cell.x, cell.y)


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------

def apply_decay(players: dict[int, Player], dt: float) -> list[tuple[int, int]]:
    """
    Apply mass decay. Returns list of (player_id, cell_id) pairs that were
    removed due to falling below MIN_CELL_MASS.
    Detached cells (merge_timer > 0) decay at DETACHED_DECAY_MULTIPLIER times the normal rate.
    """
    removed = []
    for player in players.values():
        surviving = []
        for cell in player.cells:
            if cell.mass > config.DECAY_MIN_MASS:
                multiplier = config.DETACHED_DECAY_MULTIPLIER if cell.merge_timer > 0 else 1.0
                cell.mass -= cell.mass * config.DECAY_RATE * multiplier * dt
            if cell.mass < config.MIN_CELL_MASS:
                removed.append((player.id, cell.id))
            else:
                surviving.append(cell)
        player.cells = surviving
    return removed


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def update_merge_timers(players: dict[int, Player], cell_grid: SpatialGrid, food_mgr: "FoodManager", dt: float) -> None:
    """Decrement merge timers and merge overlapping same-player cells that are ready."""
    for player in players.values():
        for cell in player.cells:
            if cell.merge_timer > 0:
                cell.merge_timer = max(-config.MERGE_PULL_MAX_TIME, cell.merge_timer - dt)

        _try_merge(player, cell_grid, food_mgr)


def apply_merge_attraction(players: dict[int, Player], cell_grid: SpatialGrid, dt: float) -> None:
    """Pull same-player cells toward each other once their merge timer has expired.
    Pull strength ramps from MERGE_PULL_BASE to MERGE_PULL_MAX over MERGE_PULL_RAMP seconds."""
    for player in players.values():
        cells = player.cells
        if len(cells) < 2:
            continue
        for i in range(len(cells)):
            a = cells[i]
            if a.merge_timer > 0:
                continue
            for j in range(i + 1, len(cells)):
                b = cells[j]
                if b.merge_timer > 0:
                    continue
                dx = b.x - a.x
                dy = b.y - a.y
                dist = math.hypot(dx, dy)
                if dist < 1.0:
                    continue
                # Time both cells have been ready (use the lesser of the two)
                time_ready = min(-a.merge_timer, -b.merge_timer)
                t = min(1.0, time_ready / config.MERGE_PULL_RAMP)
                pull = config.MERGE_PULL_BASE + (config.MERGE_PULL_MAX - config.MERGE_PULL_BASE) * t
                move = min(pull * dt, dist * 0.45)
                nx = dx / dist
                ny = dy / dist
                a.x += nx * move
                a.y += ny * move
                b.x -= nx * move
                b.y -= ny * move
                cell_grid.move(a.id, a.x, a.y)
                cell_grid.move(b.id, b.x, b.y)


def _try_merge(player: Player, cell_grid: SpatialGrid, food_mgr: "FoodManager") -> None:
    """Merge overlapping cells with merge_timer == 0."""
    if len(player.cells) < 2:
        return
    merged = True
    while merged and len(player.cells) > 1:
        merged = False
        survivors = list(player.cells)
        to_remove: set[int] = set()
        for i in range(len(survivors)):
            if survivors[i].id in to_remove:
                continue
            for j in range(i + 1, len(survivors)):
                if survivors[j].id in to_remove:
                    continue
                a, b = survivors[i], survivors[j]
                if a.merge_timer > 0 or b.merge_timer > 0:
                    continue
                dist = math.hypot(a.x - b.x, a.y - b.y)
                if dist < max(a.radius, b.radius):
                    # Merge b into a (keep the larger)
                    bigger, smaller = (a, b) if a.mass >= b.mass else (b, a)
                    # Weighted average position (smaller's mass goes to remnant food, not to bigger)
                    total = bigger.mass + smaller.mass
                    bigger.x = (bigger.x * bigger.mass + smaller.x * smaller.mass) / total
                    bigger.y = (bigger.y * bigger.mass + smaller.y * smaller.mass) / total
                    color_idx = player.id % len(config.FOOD_COLORS)
                    food_mgr.spawn_remnant(smaller.x, smaller.y, color_idx, smaller.mass)
                    cell_grid.move(bigger.id, bigger.x, bigger.y)
                    to_remove.add(smaller.id)
                    merged = True
        player.cells = [c for c in player.cells if c.id not in to_remove]


# ---------------------------------------------------------------------------
# Food collisions
# ---------------------------------------------------------------------------

def check_food_collisions(
    players: dict[int, Player],
    food_mgr: "FoodManager",
    food_grid: SpatialGrid,
) -> None:
    """Check each player cell against nearby food. Eat overlapping food."""
    for player in players.values():
        for cell in player.cells:
            r = cell.radius
            nearby_ids = food_grid.query_radius(cell.x, cell.y, r)
            for fid in list(nearby_ids):
                food = food_mgr.get(fid)
                if food is None:
                    continue
                if food.is_remnant and cell.mass <= food.mass:
                    continue
                dist = math.hypot(cell.x - food.x, cell.y - food.y)
                if dist < r:
                    mass = food_mgr.eat(fid)
                    cell.mass += mass


# ---------------------------------------------------------------------------
# Cell-vs-cell collisions
# ---------------------------------------------------------------------------

def check_cell_collisions(
    players: dict[int, Player],
    cell_grid: SpatialGrid,
    cell_map: dict[int, tuple[Cell, Player]],  # cell_id -> (cell, player)
) -> list[int]:
    """
    Handle cell-vs-cell interactions:
      - Same player, cannot merge yet: elastic push
      - Different player: eat if attacker is 1.1x larger
    Returns list of (cell_ids) that were eaten/removed.
    """
    eaten_cell_ids = []

    for player in players.values():
        for cell in list(player.cells):
            r = cell.radius
            # Query a bit larger to catch cells we might eat
            nearby_ids = cell_grid.query_radius(cell.x, cell.y, r * 2.5)
            for other_id in nearby_ids:
                if other_id == cell.id:
                    continue
                entry = cell_map.get(other_id)
                if entry is None:
                    continue
                other_cell, other_player = entry

                dx = cell.x - other_cell.x
                dy = cell.y - other_cell.y
                dist = math.hypot(dx, dy)

                if other_player.id == player.id:
                    # Same player: push apart if not ready to merge
                    if cell.merge_timer > 0 or other_cell.merge_timer > 0:
                        overlap = (cell.radius + other_cell.radius) - dist
                        if overlap > 0 and dist > 0.001:
                            push = overlap * 0.5
                            nx = dx / dist
                            ny = dy / dist
                            cell.x += nx * push * 0.5
                            cell.y += ny * push * 0.5
                            other_cell.x -= nx * push * 0.5
                            other_cell.y -= ny * push * 0.5
                            cell_grid.move(cell.id, cell.x, cell.y)
                            cell_grid.move(other_cell.id, other_cell.x, other_cell.y)
                else:
                    # Different player: eat check
                    # Attacker must be EAT_RATIO times bigger in mass
                    if cell.mass > config.EAT_RATIO * other_cell.mass:
                        # Eat if cell center is inside attacker radius
                        eat_dist = cell.radius - other_cell.radius * 0.3
                        if dist < eat_dist:
                            cell.mass += other_cell.mass
                            eaten_cell_ids.append(other_cell.id)
                            other_player.cells = [c for c in other_player.cells if c.id != other_cell.id]
                            cell_grid.remove(other_cell.id)
                            cell_map.pop(other_cell.id, None)

    return eaten_cell_ids


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------

def perform_split(
    player: Player,
    cell_grid: SpatialGrid,
    cell_map: dict[int, tuple[Cell, Player]],
    id_counter,  # itertools.count
) -> None:
    """Split each qualifying cell in two."""
    if not player.split_pending:
        return
    player.split_pending = False

    new_cells = []
    for cell in list(player.cells):
        if len(player.cells) + len(new_cells) >= config.MAX_CELLS:
            break
        if cell.mass < config.MIN_SPLIT_MASS:
            continue

        # Direction toward mouse
        dx = player.target_x - cell.x
        dy = player.target_y - cell.y
        dist = math.hypot(dx, dy) or 1.0
        nx = dx / dist
        ny = dy / dist

        half_mass = cell.mass / 2.0
        cell.mass = half_mass

        new_id = next(id_counter)
        child = Cell(
            id=new_id,
            player_id=player.id,
            x=cell.x + nx * cell.radius,
            y=cell.y + ny * cell.radius,
            mass=half_mass,
            vx=nx * config.SPLIT_SPEED,
            vy=ny * config.SPLIT_SPEED,
            merge_timer=config.MERGE_TIME_BASE,
        )
        # Recoil the parent cell opposite to the split direction
        cell.vx -= nx * config.SPLIT_SPEED * config.SPLIT_RECOIL
        cell.vy -= ny * config.SPLIT_SPEED * config.SPLIT_RECOIL
        cell.merge_timer = max(cell.merge_timer, config.MERGE_TIME_BASE)

        new_cells.append(child)
        cell_grid.insert(child.id, child.x, child.y)
        cell_map[child.id] = (child, player)

    player.cells.extend(new_cells)


def apply_split_velocity(players: dict[int, Player], cell_grid: SpatialGrid, dt: float) -> None:
    """Decelerate split ejection velocity over time."""
    for player in players.values():
        for cell in player.cells:
            if cell.merge_timer <= 0:
                continue
            speed = math.hypot(cell.vx, cell.vy)
            if speed < 1.0:
                cell.vx = 0.0
                cell.vy = 0.0
                continue
            # Decelerate
            decel = min(speed, speed * config.SPLIT_DECEL * dt)
            factor = (speed - decel) / speed
            cell.vx *= factor
            cell.vy *= factor
            cell.x = max(0, min(config.WORLD_W, cell.x + cell.vx * dt))
            cell.y = max(0, min(config.WORLD_H, cell.y + cell.vy * dt))
            cell_grid.move(cell.id, cell.x, cell.y)


# ---------------------------------------------------------------------------
# Eject
# ---------------------------------------------------------------------------

def perform_eject(
    player: Player,
    food_mgr: "FoodManager",
    food_grid: SpatialGrid,
) -> None:
    """Eject a food pellet from each cell in mouse direction."""
    if not player.eject_pending:
        return
    player.eject_pending = False

    for cell in player.cells:
        if cell.mass <= config.EJECT_MASS_COST + config.MIN_CELL_MASS:
            continue
        dx = player.target_x - cell.x
        dy = player.target_y - cell.y
        dist = math.hypot(dx, dy) or 1.0
        nx = dx / dist
        ny = dy / dist

        cell.mass -= config.EJECT_MASS_COST

        # Spawn ejected food slightly outside the cell
        ex = cell.x + nx * (cell.radius + 5)
        ey = cell.y + ny * (cell.radius + 5)
        ex = max(0, min(config.WORLD_W, ex))
        ey = max(0, min(config.WORLD_H, ey))

        # Use a random color for ejected food
        color_idx = random.randrange(len(config.FOOD_COLORS))
        food_mgr.spawn_ejected(ex, ey, color_idx, config.EJECT_MASS)
