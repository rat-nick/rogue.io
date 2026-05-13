from __future__ import annotations
import math
import random
from typing import TYPE_CHECKING

from . import config
from .player import Cell, Player
from .spatial import SpatialGrid

if TYPE_CHECKING:
    from .food import FoodManager


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def apply_input(player: Player, dt: float) -> None:
    """Move each cell toward player's target. Split cells are steerable immediately (agar.io behaviour)."""
    for cell in player.cells:
        dx = player.target_x - cell.x
        dy = player.target_y - cell.y
        dist = math.hypot(dx, dy)
        if dist < 1.0:
            continue
        nx = dx / dist
        ny = dy / dist
        spd = cell.speed
        # Don't overshoot: cap displacement to remaining distance
        move = min(spd * dt, dist)
        cell.x = max(0.0, min(config.WORLD_W, cell.x + nx * move))
        cell.y = max(0.0, min(config.WORLD_H, cell.y + ny * move))


def update_positions(cells: list[Cell], cell_grid: SpatialGrid) -> None:
    """Sync spatial grid after all cells have moved."""
    for cell in cells:
        cell_grid.move(cell.id, cell.x, cell.y)


# ---------------------------------------------------------------------------
# Decay
# ---------------------------------------------------------------------------

def apply_decay(players: dict[int, Player], dt: float) -> list[tuple[int, int]]:
    """Apply mass decay. Returns list of (player_id, cell_id) pairs removed.
    All cells decay at the same rate (no split multiplier) — matches agar.io."""
    removed = []
    for player in players.values():
        surviving = []
        for cell in player.cells:
            if cell.mass > config.DECAY_MIN_MASS:
                cell.mass -= cell.mass * config.DECAY_RATE * dt
            if cell.mass < config.MIN_CELL_MASS:
                removed.append((player.id, cell.id))
            else:
                surviving.append(cell)
        player.cells = surviving
    return removed


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def update_merge_timers(players: dict[int, Player], cell_grid: SpatialGrid, cell_map: dict, dt: float) -> None:
    """Decrement merge timers and merge overlapping same-player cells that are ready."""
    for player in players.values():
        for cell in player.cells:
            if cell.merge_timer > 0:
                cell.merge_timer = max(0.0, cell.merge_timer - dt)

        _try_merge(player, cell_grid, cell_map)


def apply_merge_attraction(players: dict[int, Player], cell_grid: SpatialGrid, dt: float) -> None:
    """No-op: agar.io cells do not magnetically attract — they converge naturally
    because all pieces chase the same cursor target."""


def _try_merge(player: Player, cell_grid: SpatialGrid, cell_map: dict) -> None:
    """Merge overlapping cells when both have merge_timer == 0.
    Mass is fully transferred to the larger cell (no remnant food) — matches agar.io."""
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
                bigger, smaller = (a, b) if a.mass >= b.mass else (b, a)
                dist = math.hypot(a.x - b.x, a.y - b.y)
                # agar.io eat distance: dist² < bigger.radius² - smaller.radius² * 0.5
                eat_dist_sq = bigger.radius ** 2 - smaller.radius ** 2 * 0.5
                if dist * dist < eat_dist_sq:
                    # Transfer all mass — no remnant food
                    bigger.mass += smaller.mass
                    cell_grid.remove(smaller.id)
                    cell_map.pop(smaller.id, None)
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
            for fid in nearby_ids:
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
# Virus collisions
# ---------------------------------------------------------------------------

def check_virus_collisions(
    players: dict[int, Player],
    virus_mgr,  # VirusManager
    virus_grid: SpatialGrid,
    cell_grid: SpatialGrid,
    cell_map: dict[int, tuple],
    food_mgr,  # FoodManager
    cell_id_counter,  # itertools.count
) -> list[int]:
    """
    Check cells against viruses. Large cells get split when hitting virus.
    Returns list of cell IDs that were removed (they got split).
    """
    removed_cell_ids = []
    
    for player in list(players.values()):
        for cell in list(player.cells):
            # Only large cells trigger virus split
            if cell.mass <= config.VIRUS_SPLIT_THRESHOLD:
                continue
            
            nearby_virus_ids = virus_grid.query_radius(cell.x, cell.y, cell.radius + config.VIRUS_RADIUS)
            for vid in nearby_virus_ids:
                virus = virus_mgr.get(vid)
                if virus is None:
                    continue
                
                dist = math.hypot(cell.x - virus.x, cell.y - virus.y)
                if dist < cell.radius + config.VIRUS_RADIUS * 0.5:
                    # Split the cell into many pieces
                    _split_cell_by_virus(player, cell, virus, cell_grid, cell_map, food_mgr, cell_id_counter)
                    removed_cell_ids.append(cell.id)
                    break  # Cell is destroyed, no more virus checks
    
    return removed_cell_ids


def _split_cell_by_virus(
    player,
    cell,
    virus,
    cell_grid: SpatialGrid,
    cell_map: dict,
    food_mgr,
    cell_id_counter,
):
    """Split a cell that hit a virus.
    agar.io behaviour: split into as many equal pieces as needed to reach MAX_CELLS,
    firing them outward in a fan. The direction of the first piece is away from the virus."""
    from .player import Cell

    # How many new pieces can we create?
    slots_available = config.MAX_CELLS - len(player.cells)
    if slots_available <= 0:
        # At cell cap: destroy the cell and scatter its mass as food
        try:
            player.cells.remove(cell)
        except ValueError:
            return
        cell_grid.remove(cell.id)
        cell_map.pop(cell.id, None)

        # Scatter mass as food pellets flung away from the virus
        n_pellets = max(1, min(int(cell.mass / 10.0), 16))
        mass_per_pellet = cell.mass / n_pellets
        dvx = cell.x - virus.x
        dvy = cell.y - virus.y
        base_angle = math.atan2(dvy, dvx)
        angle_step = 2.0 * math.pi / n_pellets
        for i in range(n_pellets):
            angle = base_angle + i * angle_step
            speed = config.EJECT_SPEED * 0.8
            food_mgr.spawn_ejected(
                x=cell.x + math.cos(angle) * 15.0,
                y=cell.y + math.sin(angle) * 15.0,
                color_idx=0,
                mass=mass_per_pellet,
                vx=math.cos(angle) * speed,
                vy=math.sin(angle) * speed,
            )
        return

    # We replace the current cell with (slots_available + 1) equal pieces
    # (+1 because the original cell itself becomes one piece)
    num_pieces = slots_available + 1
    mass_per_piece = cell.mass / num_pieces

    # Remove the original cell (guard in case it was already removed this tick)
    try:
        player.cells.remove(cell)
    except ValueError:
        return
    cell_grid.remove(cell.id)
    cell_map.pop(cell.id, None)

    # Direction of first piece: away from virus centre
    dvx = cell.x - virus.x
    dvy = cell.y - virus.y
    base_angle = math.atan2(dvy, dvx)

    merge_time = config.MERGE_TIME_BASE + config.MERGE_TIME_MASS_FACTOR * mass_per_piece

    angle_step = 2.0 * math.pi / num_pieces
    for i in range(num_pieces):
        angle = base_angle + i * angle_step
        cx = math.cos(angle)
        cy = math.sin(angle)

        new_id = next(cell_id_counter)
        new_cell = Cell(
            id=new_id,
            player_id=player.id,
            x=cell.x + cx * 10.0,
            y=cell.y + cy * 10.0,
            mass=mass_per_piece,
            vx=cx * config.SPLIT_SPEED,
            vy=cy * config.SPLIT_SPEED,
            merge_timer=merge_time,
            collision_restore_ticks=config.COLLISION_RESTORE_TICKS,
        )
        player.cells.append(new_cell)
        cell_grid.insert(new_cell.id, new_cell.x, new_cell.y)
        cell_map[new_cell.id] = (new_cell, player)


def check_ejected_virus_feeding(
    ejected_food_ids: list[int],
    food_mgr,
    virus_mgr,
    virus_grid: SpatialGrid,
) -> None:
    """Check if ejected mass hits viruses and feed them."""
    for fid in ejected_food_ids:
        food = food_mgr.get(fid)
        if food is None:
            continue
        
        # Check nearby viruses
        nearby_virus_ids = virus_grid.query_radius(food.x, food.y, config.VIRUS_RADIUS)
        for vid in nearby_virus_ids:
            virus = virus_mgr.get(vid)
            if virus is None:
                continue
            
            dist = math.hypot(food.x - virus.x, food.y - virus.y)
            if dist < config.VIRUS_RADIUS:
                # Feed the virus
                food_mass = food_mgr.eat(fid)
                if food_mass > 0:
                    should_shoot = virus_mgr.feed(vid, food_mass)
                    if should_shoot:
                        # Calculate direction from food velocity or random
                        direction_x = food.vx if food.vx != 0 else 1.0
                        direction_y = food.vy if food.vy != 0 else 0.0
                        virus_mgr.shoot_virus(vid, direction_x, direction_y)
                break  # Food consumed


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
      - Same player, cannot merge yet: elastic push using Ogar's proportional formula
        (skip if either cell is in collision_restore_ticks window after a split)
      - Different player: eat if attacker has EAT_RATIO × more mass and center is inside
        the agar.io engulf radius: dist² < attacker_r² - target_r² * 0.5
    Returns list of cell IDs that were eaten/removed.
    """
    eaten_cell_ids = []

    for player in players.values():
        for cell in list(player.cells):
            r = cell.radius
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
                    # Same player: push apart when not ready to merge,
                    # but only after the collision-restore window expires.
                    if cell.merge_timer > 0 or other_cell.merge_timer > 0:
                        if cell.collision_restore_ticks > 0 or other_cell.collision_restore_ticks > 0:
                            continue  # just-split cells pass through each other briefly
                        overlap = (r + other_cell.radius) - dist
                        if overlap > 0 and dist > 0.001:
                            # Ogar formula: each cell moves proportionally to the other's size
                            max_dist = r + other_cell.radius
                            move1 = overlap * (max_dist / r) * 0.25
                            move2 = overlap * (max_dist / other_cell.radius) * 0.25
                            nx = dx / dist
                            ny = dy / dist
                            cell.x += nx * move1
                            cell.y += ny * move1
                            other_cell.x -= nx * move2
                            other_cell.y -= ny * move2
                            cell_grid.move(cell.id, cell.x, cell.y)
                            cell_grid.move(other_cell.id, other_cell.x, other_cell.y)
                else:
                    # Different player: eat if sufficiently larger and close enough
                    if cell.mass >= config.EAT_RATIO * other_cell.mass:
                        # agar.io engulf condition: dist² < attacker_r² - target_r² * 0.5
                        eat_dist_sq = r * r - other_cell.radius * other_cell.radius * 0.5
                        if dist * dist < eat_dist_sq:
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
    """Split each qualifying cell in two — exactly as agar.io/Ogar.
    No recoil on parent. Both parent and child get full merge timer based on half-mass.
    Both get collision_restore_ticks to pass through each other briefly."""
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

        # agar.io merge timer: MERGE_TIME_BASE + MERGE_TIME_MASS_FACTOR * mass_at_split
        merge_time = config.MERGE_TIME_BASE + config.MERGE_TIME_MASS_FACTOR * half_mass
        cell.merge_timer = max(cell.merge_timer, merge_time)
        cell.collision_restore_ticks = config.COLLISION_RESTORE_TICKS

        new_id = next(id_counter)
        child = Cell(
            id=new_id,
            player_id=player.id,
            x=cell.x + nx * cell.radius,
            y=cell.y + ny * cell.radius,
            mass=half_mass,
            vx=nx * config.SPLIT_SPEED,
            vy=ny * config.SPLIT_SPEED,
            merge_timer=merge_time,
            collision_restore_ticks=config.COLLISION_RESTORE_TICKS,
        )

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


def update_collision_restore_ticks(players: dict[int, Player]) -> None:
    """Decrement per-cell collision-restore counters each tick."""
    for player in players.values():
        for cell in player.cells:
            if cell.collision_restore_ticks > 0:
                cell.collision_restore_ticks -= 1


# ---------------------------------------------------------------------------
# Eject
# ---------------------------------------------------------------------------

def perform_eject(
    player: Player,
    food_mgr: "FoodManager",
    food_grid: SpatialGrid,
) -> list[int]:
    """
    Eject a food pellet from each cell in mouse direction (agar.io / Ogar).
    Cell must have mass >= EJECT_MIN_MASS (playerMinMassEject = 32).
    Cost: EJECT_MASS_COST (15) deducted; pellet spawned with EJECT_MASS (13).
    Returns list of food IDs for ejected mass (to check virus feeding).
    """
    if not player.eject_pending:
        return []
    player.eject_pending = False

    ejected_ids = []
    for cell in player.cells:
        if cell.mass < config.EJECT_MIN_MASS:
            continue
        dx = player.target_x - cell.x
        dy = player.target_y - cell.y
        dist = math.hypot(dx, dy) or 1.0
        nx = dx / dist
        ny = dy / dist

        cell.mass -= config.EJECT_MASS_COST

        # Spawn ejected food just outside the cell; add slight random spread (±0.3 rad) like Ogar
        spread = (random.random() - 0.5) * 0.6
        ex_angle = math.atan2(ny, nx) + spread
        enx = math.cos(ex_angle)
        eny = math.sin(ex_angle)

        ex = cell.x + enx * (cell.radius + 16)
        ey = cell.y + eny * (cell.radius + 16)
        ex = max(0, min(config.WORLD_W, ex))
        ey = max(0, min(config.WORLD_H, ey))

        vx = enx * config.EJECT_SPEED
        vy = eny * config.EJECT_SPEED

        color_idx = random.randrange(len(config.FOOD_COLORS))
        fid = food_mgr.spawn_ejected(ex, ey, color_idx, config.EJECT_MASS, vx, vy)
        ejected_ids.append(fid)
    
    return ejected_ids
