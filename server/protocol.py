from __future__ import annotations
import struct
from typing import Any

import msgpack

from .food import Food

# Message type constants
MSG_INIT = 0x10
MSG_TICK = 0x11
MSG_DEAD = 0x12

# Manual next-generation trigger (client -> server)
MSG_NEXT_GEN = 0x25
# Toggle early-next-gen flag (client -> server): [MSG_SET_EARLY_NEXT_GEN, bool]
MSG_SET_EARLY_NEXT_GEN = 0x26
# Set simulation time scale (client -> server): [MSG_SET_TIME_SCALE, float]
MSG_SET_TIME_SCALE = 0x27

# Client -> Server input format: !BffB (10 bytes)
# B = msg_type (0x01)
# f = mouse_world_x
# f = mouse_world_y
# B = flags (bit0=split, bit1=eject)
_INPUT_STRUCT = struct.Struct('!BffB')
MSG_INPUT = 0x01


def decode_input(data: bytes) -> dict | None:
    """
    Decode a client input packet.
    Returns dict with keys: msg_type, mouse_x, mouse_y, split, eject
    Returns None if packet is malformed.
    """
    if len(data) < _INPUT_STRUCT.size:
        return None
    try:
        msg_type, mx, my, flags = _INPUT_STRUCT.unpack_from(data)
    except struct.error:
        return None
    if msg_type != MSG_INPUT:
        return None
    # Validate: discard NaN/inf positions
    if not (0 <= mx <= 100000 and 0 <= my <= 100000):
        # Accept values outside world, server will clamp mouse position
        pass
    return {
        'msg_type': msg_type,
        'mouse_x': mx,
        'mouse_y': my,
        'split': bool(flags & 0x01),
        'eject': bool(flags & 0x02),
    }


def encode_init(player_id: int, world_w: int, world_h: int, tick_rate: int) -> bytes:
    """Encode the initial handshake packet."""
    return msgpack.packb([MSG_INIT, player_id, world_w, world_h, tick_rate], use_bin_type=True)


def encode_tick(
    tick_num: int,
    own_cell_ids: list[int],
    visible_cells: list,        # list of Cell objects visible to this player
    food_new: list,             # Food objects spawned this tick visible to player
    food_removed: list[int],    # food IDs eaten this tick (all clients remove these)
    virus_new: list,            # Virus objects spawned this tick
    virus_removed: list[int],   # virus IDs removed this tick
    known_player_ids: set[int], # player IDs whose names this client already knows
    player_map: dict[int, Any], # player_id -> Player (for name lookup)
    leaderboard: list | None,   # [[name, score], ...] or None
) -> bytes:
    """
    Encode a tick update packet.
    Cell format: [id, x, y, mass, player_id, name_or_null]
    name is null if the client already received this player's name.
    """
    cells_out = []
    for c in visible_cells:
        pid = c.player_id
        if pid not in known_player_ids:
            known_player_ids.add(pid)
            p = player_map.get(pid)
            name = p.name if p else ''
            hue  = p.hue  if p else -1
        else:
            name = None
            hue  = None
        cells_out.append([c.id, round(c.x, 1), round(c.y, 1), round(c.mass, 2), pid, name, hue])

    food_out = [[f.id, round(f.x, 1), round(f.y, 1), f.color_idx, round(f.mass, 1), round(f.vx, 1), round(f.vy, 1)] for f in food_new]
    
    virus_out = [[v.id, round(v.x, 1), round(v.y, 1), round(v.mass, 1)] for v in virus_new]

    lb_out = [[name, score] for name, score in leaderboard] if leaderboard is not None else None

    payload = [MSG_TICK, tick_num, own_cell_ids, cells_out, food_out, food_removed, virus_out, virus_removed, lb_out]
    return msgpack.packb(payload, use_bin_type=True)


def encode_dead(final_score: int, killer_name: str) -> bytes:
    """Encode a death notification packet."""
    return msgpack.packb([MSG_DEAD, final_score, killer_name], use_bin_type=True)


# ---------------------------------------------------------------------------
# Spectator protocol
# ---------------------------------------------------------------------------

MSG_STATS  = 0x22  # server -> spectator: all player stats
MSG_FOLLOW = 0x23  # client -> server: which player_id to follow + zoom_idx

# Follow packet: B=msg_type, I=player_id (0=overview), B=zoom_idx  (6 bytes total)
_FOLLOW_STRUCT = struct.Struct('!BIB')

# Viewport size multipliers indexed by zoom_idx.
# Higher value = more zoomed IN (smaller visible world area).
ZOOM_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.5, 4.0]
ZOOM_LABELS = ['0.25×', '0.5×', '0.75×', '1×', '1.5×', '2.5×', '4×']


def decode_follow(data: bytes) -> dict | None:
    """Decode a spectator follow/zoom command. Returns None if malformed."""
    if len(data) < _FOLLOW_STRUCT.size:
        return None
    try:
        msg_type, pid, zoom_idx = _FOLLOW_STRUCT.unpack_from(data)
    except struct.error:
        return None
    if msg_type != MSG_FOLLOW:
        return None
    return {
        'player_id': pid,  # 0 = overview / unfollow
        'zoom_idx': min(int(zoom_idx), len(ZOOM_MULTIPLIERS) - 1),
    }


def encode_stats(
    tick_num: int,
    players_info: list,  # [[id, name, mass, cx, cy, cell_count, is_bot], ...]
    total_food: int,
) -> bytes:
    """Encode a stats snapshot for spectators."""
    return msgpack.packb([MSG_STATS, tick_num, players_info, total_food], use_bin_type=True)


# Training-mode stats packet
MSG_TRAINING_STATS = 0x24  # server -> training viewer


def encode_training_stats(
    tick_num: int,
    generation: int,
    time_remaining: float,
    pop_size: int,
    top_fitness: float,
    avg_fitness: float,
    best_mass: float,
    avg_mass: float,
    total_deaths: int,
    players_info: list,  # same format as encode_stats: [[id, name, mass, cx, cy, cell_count, is_bot], ...]
    total_food: int,
    early_next_gen: bool = True,
    time_scale: float = 1.0,
) -> bytes:
    """Encode a training-mode stats packet for the training viewer."""
    return msgpack.packb(
        [
            MSG_TRAINING_STATS,
            tick_num,
            generation,
            round(time_remaining, 1),
            pop_size,
            round(top_fitness, 1),
            round(avg_fitness, 1),
            round(best_mass, 1),
            round(avg_mass, 1),
            total_deaths,
            players_info,
            total_food,
            1 if early_next_gen else 0,  # msg[12]
            round(time_scale, 2),         # msg[13]
        ],
        use_bin_type=True,
    )
