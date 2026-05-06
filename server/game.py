from __future__ import annotations

import asyncio
import itertools
import logging
import math
import random
from typing import Any

from . import config
from . import physics
from . import protocol
from .bot import BotController, NullWebSocket
from .genetics import GenomePool, random_genome
from .food import FoodManager
from .player import Cell, Player
from .spatial import SpatialGrid

logger = logging.getLogger(__name__)

_BOT_NAMES = [
    "Amoeba", "Globule", "Nucleus", "Cytoplasm", "Vacuole",
    "Mitosis", "Osmosis", "Plasmid", "Ribosome", "Flagella",
    "Dendrite", "Axon", "Synapse", "Neuron", "Ganglion",
    "Spore", "Zygote", "Gamete", "Protist", "Archaea",
]
_bot_name_idx = 0


def _random_bot_name() -> str:
    global _bot_name_idx
    name = _BOT_NAMES[_bot_name_idx % len(_BOT_NAMES)]
    _bot_name_idx += 1
    return name


class GameWorld:
    def __init__(self) -> None:
        # ID counters
        self._player_id_counter = itertools.count(1)
        self._cell_id_counter   = itertools.count(1)

        # Players and cells
        self.players: dict[int, Player] = {}
        # Flat lookup: cell_id -> (Cell, Player) — for O(1) collision resolution
        self.cell_map: dict[int, tuple[Cell, Player]] = {}

        # Spatial grids
        self.cell_grid = SpatialGrid()
        self.food_grid = SpatialGrid()

        # Food manager
        self.food_mgr = FoodManager(self.food_grid)

        # Tick counter
        self.tick_counter = 0

        # Leaderboard cache (rebuilt every LEADERBOARD_INTERVAL ticks)
        self._leaderboard: list[tuple[str, int]] = []

        # Bots
        self._bot_ids: set[int] = set()
        self._bot_controller = BotController()
        # Genetic algorithm
        self._genome_pool: GenomePool = GenomePool.load(config.NEAT_SAVE_PATH)
        self._bot_death_count: int = 0

        # Spectators: spec_id -> {ws, follow_id, zoom_idx, known_player_ids, sent_food_ids}
        self._spectators: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Public API called from main.py
    # ------------------------------------------------------------------

    def seed_food(self) -> None:
        """Spawn initial food. Call once before starting the tick loop."""
        self.food_mgr.spawn_batch(config.FOOD_TARGET)
        # Flush delta so this initial food is not sent as "new this tick"
        self.food_mgr.flush_delta()

    def seed_bots(self) -> None:
        """Spawn BOT_COUNT bots. Uses best genome from pool if available."""
        best = self._genome_pool.best(1)
        best_genome = best[0][1] if best else None
        if best_genome is not None:
            logger.info(
                f"Seeding bots with best genome (fitness={best[0][0]:.1f}, "
                f"gen={self._genome_pool.generation})"
            )
        for _ in range(config.BOT_START):
            self._spawn_bot(genome=best_genome)

    async def handle_connection(self, websocket) -> None:
        """Entry point for each WebSocket connection."""
        player_id = next(self._player_id_counter)
        name = "Player"  # default; overridden by first message if it's a join msg

        # Expect first message to be player name (plain text) or input packet
        # We'll read the first message with a short timeout to get the name
        try:
            first_msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            first_msg = b''

        if isinstance(first_msg, str):
            # Spectator handshake
            if first_msg.strip().upper() == 'SPECTATE':
                logger.info("Spectator connected")
                await self._handle_spectator_connection(websocket)
                return
            name = first_msg.strip()[:16] or "Player"
        # If bytes, it will be processed as input below (player uses default name)

        player = self._spawn_player_replacing_bot(player_id, name, websocket)
        logger.info(f"Player {player_id} '{name}' connected")

        try:
            # Send init packet
            init_pkt = protocol.encode_init(
                player_id, config.WORLD_W, config.WORLD_H, config.TICK_RATE
            )
            await websocket.send(init_pkt)

            # Process input messages
            async for raw in websocket:
                if isinstance(raw, bytes):
                    inp = protocol.decode_input(raw)
                    if inp is None:
                        continue
                    # Server-side clamping of mouse position (security: don't trust client)
                    player.target_x = max(0.0, min(float(config.WORLD_W), inp['mouse_x']))
                    player.target_y = max(0.0, min(float(config.WORLD_H), inp['mouse_y']))
                    if inp['split']:  player.split_pending = True
                    if inp['eject']: player.eject_pending  = True

        except Exception as exc:
            logger.debug(f"Player {player_id} connection error: {exc}")
        finally:
            self._remove_player(player_id)
            self._spawn_bot()
            logger.info(f"Player {player_id} '{name}' disconnected")

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------

    async def tick_loop(self) -> None:
        """Main game loop. Runs at TICK_RATE ticks per second."""
        loop = asyncio.get_event_loop()
        tick_interval = config.TICK_INTERVAL
        next_tick = loop.time()

        while True:
            next_tick += tick_interval
            await self._tick()
            now = loop.time()
            sleep_time = next_tick - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                # We're behind; skip sleep but don't accumulate debt indefinitely
                next_tick = loop.time()

    async def _tick(self) -> None:
        dt = config.TICK_INTERVAL
        self.tick_counter += 1

        if not self.players:
            # Maintain food level even with no players
            deficit = config.FOOD_TARGET - self.food_mgr.count()
            if deficit > 0:
                self.food_mgr.spawn_batch(min(deficit, 50))
            self.food_mgr.flush_delta()
            return

        # ---- Bot AI ----
        self._bot_controller.update(self, dt)

        # ---- Physics ----
        for player in list(self.players.values()):
            physics.apply_input(player, dt)

        physics.apply_split_velocity(self.players, self.cell_grid, dt)

        # Sync all moved cells into spatial grid
        all_cells = [c for p in self.players.values() for c in p.cells]
        physics.update_positions(all_cells, self.cell_grid)

        decayed = physics.apply_decay(self.players, dt)
        for _pid, cell_id in decayed:
            self.cell_grid.remove(cell_id)
            self.cell_map.pop(cell_id, None)
        physics.apply_merge_attraction(self.players, self.cell_grid, dt)
        physics.update_merge_timers(self.players, self.cell_grid, self.food_mgr, dt)
        self.food_mgr.tick_decay(dt)

        physics.check_food_collisions(self.players, self.food_mgr, self.food_grid)

        eaten_ids = physics.check_cell_collisions(self.players, self.cell_grid, self.cell_map)
        for cell_id in eaten_ids:
            self.cell_map.pop(cell_id, None)

        # Handle splits and ejects
        for player in list(self.players.values()):
            physics.perform_split(player, self.cell_grid, self.cell_map, self._cell_id_counter)
            physics.perform_eject(player, self.food_mgr, self.food_grid)

        # Respawn dead players (no cells left)
        dead_players = [p for p in self.players.values() if not p.cells]
        for player in dead_players:
            if player.id in self._bot_ids:
                # Record fitness, breed new genome, respawn
                genome, fitness = self._bot_controller.unregister(player.id)
                if genome is not None:
                    self._genome_pool.add(genome, fitness)
                    self._bot_death_count += 1
                    if self._bot_death_count % 50 == 0:
                        self._genome_pool.save(config.NEAT_SAVE_PATH)
                        logger.info(
                            f"GA: gen={self._genome_pool.generation} "
                            f"deaths={self._genome_pool.total_deaths} "
                            f"pool_size={len(self._genome_pool._pool)}"
                        )
                new_genome = self._genome_pool.breed()
                self._respawn_player(player)
                cx, cy = player.centroid
                self._bot_controller.register(player.id, cx, cy, new_genome)
            else:
                await self._send_dead(player, score=0, killer_name="")
                self._respawn_player(player)

        # Food replenishment
        deficit = config.FOOD_TARGET - self.food_mgr.count()
        if deficit > 0:
            self.food_mgr.spawn_batch(min(deficit, 50))

        # Build leaderboard every N ticks
        leaderboard = None
        if self.tick_counter % config.LEADERBOARD_INTERVAL == 0:
            self._leaderboard = sorted(
                [(p.name, p.score) for p in self.players.values()],
                key=lambda x: -x[1]
            )[:10]
            leaderboard = self._leaderboard

        # Get food delta ONCE
        food_new, food_removed = self.food_mgr.flush_delta()

        # ---- Send tick packets ----
        send_tasks = []
        for player in list(self.players.values()):
            if player.id in self._bot_ids:
                continue  # bots don't receive packets
            pkt = self._build_tick_packet(player, food_removed, leaderboard)
            send_tasks.append(self._safe_send(player.websocket, pkt))

        # ---- Send to spectators ----
        if self._spectators:
            stats_pkt = None
            if self.tick_counter % config.LEADERBOARD_INTERVAL == 0:
                stats_pkt = self._build_stats_packet()
            for spec in list(self._spectators.values()):
                tick_pkt = self._build_spectator_tick_packet(spec, food_removed, leaderboard)
                send_tasks.append(self._safe_send(spec['ws'], tick_pkt))
                if stats_pkt is not None:
                    send_tasks.append(self._safe_send(spec['ws'], stats_pkt))

        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)

    # ------------------------------------------------------------------
    # Packet building
    # ------------------------------------------------------------------

    def _build_tick_packet(
        self,
        player: Player,
        food_removed_all: list[int],
        leaderboard: list | None,
    ) -> bytes:
        # Viewport culling
        vx, vy, vw, vh = player.viewport_rect

        # Visible cells
        visible_cell_ids = self.cell_grid.query_rect(vx, vy, vw, vh)
        visible_cells = [
            self.cell_map[cid][0]
            for cid in visible_cell_ids
            if cid in self.cell_map
        ]

        # Send all food in viewport that this client hasn't seen yet.
        # This covers seeded/existing food, not just food spawned this tick.
        food_new_visible = []
        in_viewport_ids = self.food_grid.query_rect(vx, vy, vw, vh)
        for fid in in_viewport_ids:
            if fid not in player.sent_food_ids:
                f = self.food_mgr.get(fid)
                if f is not None:
                    player.sent_food_ids.add(fid)
                    food_new_visible.append(f)

        own_cell_ids = [c.id for c in player.cells]

        return protocol.encode_tick(
            tick_num=self.tick_counter,
            own_cell_ids=own_cell_ids,
            visible_cells=visible_cells,
            food_new=food_new_visible,
            food_removed=food_removed_all,  # all clients must remove these
            known_player_ids=player.known_player_ids,
            player_map=self.players,
            leaderboard=leaderboard,
        )

    # ------------------------------------------------------------------
    # Spectator support
    # ------------------------------------------------------------------

    async def _handle_spectator_connection(self, websocket) -> None:
        """Handle a spectator WebSocket connection."""
        spec_id = next(self._player_id_counter)
        spec: dict = {
            'ws': websocket,
            'follow_id': None,      # None = overview
            'zoom_idx': 3,          # default 1× zoom
            'known_player_ids': set(),
            'sent_food_ids': set(),
        }
        self._spectators[spec_id] = spec
        try:
            # Send init so the client knows world size / tick rate
            init_pkt = protocol.encode_init(spec_id, config.WORLD_W, config.WORLD_H, config.TICK_RATE)
            await websocket.send(init_pkt)
            # Immediately push current stats so the sidebar populates fast
            await websocket.send(self._build_stats_packet())

            # Listen for follow/zoom commands
            async for raw in websocket:
                if isinstance(raw, bytes):
                    follow_data = protocol.decode_follow(raw)
                    if follow_data:
                        new_pid = follow_data['player_id'] or None  # 0 → None (overview)
                        if new_pid != spec['follow_id']:
                            # Reset delta caches when switching targets
                            spec['known_player_ids'].clear()
                            spec['sent_food_ids'].clear()
                        spec['follow_id'] = new_pid
                        spec['zoom_idx'] = follow_data['zoom_idx']
        except Exception as exc:
            logger.debug(f"Spectator {spec_id} error: {exc}")
        finally:
            self._spectators.pop(spec_id, None)
            logger.info(f"Spectator {spec_id} disconnected")

    def _build_spectator_tick_packet(
        self,
        spec: dict,
        food_removed_all: list[int],
        leaderboard: list | None,
    ) -> bytes:
        """Build a tick packet for a spectator, culled to their current view."""
        follow_id = spec['follow_id']
        zoom_mult = protocol.ZOOM_MULTIPLIERS[spec['zoom_idx']]

        if follow_id is not None and follow_id in self.players:
            followed = self.players[follow_id]
            cx, cy = followed.centroid
            total_mass = max(followed.total_mass, 1.0)
            own_cell_ids = [c.id for c in followed.cells]
            # Compute viewport matching player.viewport_rect, scaled by zoom and 2× buffer
            vw_base = config.VIEW_BASE_SIZE * ((total_mass / 100.0) ** config.VIEW_MASS_SCALE)
            vw = (vw_base / zoom_mult) * 2.0
            vh = vw * 9.0 / 16.0
            vx = cx - vw / 2.0
            vy = cy - vh / 2.0
            # Food delta for this viewport
            food_new_visible = []
            for fid in self.food_grid.query_rect(vx, vy, vw, vh):
                if fid not in spec['sent_food_ids']:
                    f = self.food_mgr.get(fid)
                    if f is not None:
                        spec['sent_food_ids'].add(fid)
                        food_new_visible.append(f)
        else:
            # Overview: show entire world, skip food (too much data at once)
            vx, vy = 0.0, 0.0
            vw, vh = float(config.WORLD_W), float(config.WORLD_H)
            own_cell_ids = []
            food_new_visible = []

        visible_cell_ids = self.cell_grid.query_rect(vx, vy, vw, vh)
        visible_cells = [
            self.cell_map[cid][0]
            for cid in visible_cell_ids
            if cid in self.cell_map
        ]

        return protocol.encode_tick(
            tick_num=self.tick_counter,
            own_cell_ids=own_cell_ids,
            visible_cells=visible_cells,
            food_new=food_new_visible,
            food_removed=food_removed_all,
            known_player_ids=spec['known_player_ids'],
            player_map=self.players,
            leaderboard=leaderboard,
        )

    def _build_stats_packet(self) -> bytes:
        """Build a full player-stats snapshot for spectators."""
        players_info = []
        for pid, p in self.players.items():
            cx, cy = p.centroid
            players_info.append([
                pid,
                p.name,
                round(p.total_mass, 1),
                round(cx),
                round(cy),
                len(p.cells),
                1 if pid in self._bot_ids else 0,
            ])
        return protocol.encode_stats(self.tick_counter, players_info, self.food_mgr.count())

    # ------------------------------------------------------------------
    # Player lifecycle
    # ------------------------------------------------------------------

    def _spawn_bot(self, genome=None) -> None:
        """Create a new bot player. If genome is None, breeds one from the pool."""
        if genome is None:
            genome = self._genome_pool.breed()
        pid = next(self._player_id_counter)
        name = _random_bot_name()
        bot = Player(id=pid, name=name, websocket=NullWebSocket())
        self.players[pid] = bot
        self._bot_ids.add(pid)
        x = random.uniform(config.WORLD_W * 0.05, config.WORLD_W * 0.95)
        y = random.uniform(config.WORLD_H * 0.05, config.WORLD_H * 0.95)
        self._create_cell(bot, x, y, mass=20.0)
        self._bot_controller.register(pid, x, y, genome)

    def _spawn_player_replacing_bot(self, player_id: int, name: str, websocket) -> Player:
        """Spawn a real player, removing a random bot to keep population stable."""
        if self._bot_ids:
            bot_id = random.choice(list(self._bot_ids))
            self._bot_ids.discard(bot_id)
            self._bot_controller.unregister(bot_id)  # discard fitness (forced removal)
            self._remove_player(bot_id)

        player = Player(id=player_id, name=name, websocket=websocket)
        self.players[player_id] = player
        x = random.uniform(config.WORLD_W * 0.1, config.WORLD_W * 0.9)
        y = random.uniform(config.WORLD_H * 0.1, config.WORLD_H * 0.9)
        self._create_cell(player, x, y, mass=20.0)
        return player

    def _spawn_player(self, player_id: int, name: str, websocket) -> Player:
        player = Player(id=player_id, name=name, websocket=websocket)
        self.players[player_id] = player
        self._create_cell(player, config.WORLD_W / 2, config.WORLD_H / 2, mass=20.0)
        return player

    def _respawn_player(self, player: Player) -> None:
        """Respawn an existing player with a fresh cell at a random position."""
        player.known_player_ids.clear()
        player.sent_food_ids.clear()
        x = random.uniform(config.WORLD_W * 0.1, config.WORLD_W * 0.9)
        y = random.uniform(config.WORLD_H * 0.1, config.WORLD_H * 0.9)
        self._create_cell(player, x, y, mass=20.0)

    def _create_cell(self, player: Player, x: float, y: float, mass: float) -> Cell:
        cell_id = next(self._cell_id_counter)
        cell = Cell(id=cell_id, player_id=player.id, x=x, y=y, mass=mass)
        player.cells.append(cell)
        self.cell_grid.insert(cell_id, x, y)
        self.cell_map[cell_id] = (cell, player)
        return cell

    def _remove_player(self, player_id: int) -> None:
        player = self.players.pop(player_id, None)
        if player is None:
            return
        self._bot_ids.discard(player_id)
        # Note: bot_controller.unregister is called BEFORE _remove_player in the
        # GA flow so fitness is recorded. Here we just clean up any leftover state.
        self._bot_controller.unregister(player_id)
        for cell in player.cells:
            self.cell_grid.remove(cell.id)
            self.cell_map.pop(cell.id, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _safe_send(websocket, data: bytes) -> None:
        try:
            await websocket.send(data)
        except Exception:
            pass

    async def _send_dead(self, player: Player, score: int, killer_name: str) -> None:
        pkt = protocol.encode_dead(score, killer_name)
        await self._safe_send(player.websocket, pkt)
