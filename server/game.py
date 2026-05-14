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
from .genetics import GenomePool, random_genome, genome_hue
from .food import FoodManager
from .player import Cell, Player
from .spatial import SpatialGrid
from .virus import VirusManager

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
    def __init__(self, bot_mode: str = 'neat') -> None:
        self._bot_mode = bot_mode  # 'neat' | 'ppo'

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
        self.virus_grid = SpatialGrid()

        # Food manager
        self.food_mgr = FoodManager(self.food_grid)
        
        # Virus manager
        self.virus_mgr = VirusManager(self.virus_grid)

        # Tick counter
        self.tick_counter = 0

        # Leaderboard cache (rebuilt every LEADERBOARD_INTERVAL ticks)
        self._leaderboard: list[tuple[str, int]] = []

        # Bots
        self._bot_ids: set[int] = set()
        if bot_mode == 'ppo':
            from .ppo_agent import NumpyPolicy
            from .ppo_bot import PPOBotController
            from .ppo_train import PPO_SAVE_PATH
            n_bots = config.BOT_START
            try:
                model = NumpyPolicy.from_checkpoint(PPO_SAVE_PATH)
                logger.info(f"[PPO] Loaded numpy checkpoint from {PPO_SAVE_PATH}")
            except Exception:
                # Fall back to a random ActorCritic (requires torch)
                from .ppo_agent import ActorCritic, N_OBS
                import torch
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                model = ActorCritic(n_obs=N_OBS).to(device)
                logger.info("[PPO] No checkpoint found — starting fresh policy (torch)")
            self._bot_controller = PPOBotController(
                n_bots=n_bots, model=model, inference_only=True,
            )
            self._genome_pool = None
            self._bot_death_count = 0
            self._burst_elapsed   = 0.0
        else:
            self._bot_controller = BotController()
            # Genetic algorithm
            self._genome_pool: GenomePool = GenomePool.load(config.NEAT_SAVE_PATH)
            self._bot_death_count: int = 0
            self._burst_elapsed: float = 0.0  # seconds since last bot burst

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

    def seed_viruses(self) -> None:
        """Spawn initial viruses at random positions."""
        self.virus_mgr.respawn_to_target(config.VIRUS_TARGET)
        self.virus_mgr.flush_delta()

    def seed_bots(self) -> None:
        """Spawn BOT_COUNT bots. Uses best genome from pool if available (NEAT only)."""
        if self._bot_mode == 'ppo':
            for _ in range(config.BOT_START):
                self._spawn_bot()
            return
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
                await asyncio.sleep(0)  # yield to event loop so connections can be accepted

    async def _tick(self) -> None:
        dt = config.TICK_INTERVAL
        self.tick_counter += 1

        if not self.players:
            # Maintain food level even with no players
            deficit = config.FOOD_TARGET - self.food_mgr.count()
            if deficit > 0:
                self.food_mgr.spawn_batch(deficit)
            self.food_mgr.flush_delta()
            return

        # ---- Bot AI ----
        try:
            self._bot_controller.update(self, dt)
        except Exception as exc:
            logger.warning(f"Bot controller error: {exc}")

        # ---- PPO rollout update (training world only — skipped in game server) ----
        if self._bot_mode == 'ppo' and self._bot_controller.is_rollout_full():
            stats = await asyncio.to_thread(self._do_ppo_update)
            self._bot_controller.reset_rollout()
            logger.info(
                f"[PPO] rollout complete | "
                f"pg={stats.get('pg_loss',0):.4f} "
                f"vf={stats.get('vf_loss',0):.4f} "
                f"ent={stats.get('entropy',0):.4f}"
            )

        # ---- Physics ----
        for player in list(self.players.values()):
            physics.apply_input(player, dt)

        physics.apply_split_velocity(self.players, self.cell_grid, dt)
        physics.update_collision_restore_ticks(self.players)

        # Sync all moved cells into spatial grid
        all_cells = [c for p in self.players.values() for c in p.cells]
        physics.update_positions(all_cells, self.cell_grid)

        decayed = physics.apply_decay(self.players, dt)
        for _pid, cell_id in decayed:
            self.cell_grid.remove(cell_id)
            self.cell_map.pop(cell_id, None)
        physics.apply_merge_attraction(self.players, self.cell_grid, dt)
        physics.update_merge_timers(self.players, self.cell_grid, self.cell_map, dt)
        self.food_mgr.tick_decay(dt)

        physics.check_food_collisions(self.players, self.food_mgr, self.food_grid)
        
        # Check virus collisions
        virus_split_ids = physics.check_virus_collisions(
            self.players, self.virus_mgr, self.virus_grid,
            self.cell_grid, self.cell_map, self.food_mgr, self._cell_id_counter
        )
        for cell_id in virus_split_ids:
            self.cell_map.pop(cell_id, None)

        eaten_ids = physics.check_cell_collisions(self.players, self.cell_grid, self.cell_map)
        for cell_id in eaten_ids:
            self.cell_map.pop(cell_id, None)

        # Handle splits and ejects
        for player in list(self.players.values()):
            physics.perform_split(player, self.cell_grid, self.cell_map, self._cell_id_counter)
            ejected_ids = physics.perform_eject(player, self.food_mgr, self.food_grid)
            # Check if ejected mass feeds viruses
            if ejected_ids:
                physics.check_ejected_virus_feeding(ejected_ids, self.food_mgr, self.virus_mgr, self.virus_grid)

        # Handle dead players (no cells left)
        dead_players = [p for p in self.players.values() if not p.cells]
        for player in dead_players:
            if player.id in self._bot_ids:
                if self._bot_mode == 'ppo':
                    # PPO: just respawn immediately, no genome pool
                    self._bot_controller.unregister(player.id)
                    self._bot_ids.discard(player.id)
                    self.players.pop(player.id, None)
                    self._spawn_bot()
                else:
                    # NEAT: record fitness, add genome to pool
                    bs = self._bot_controller._state.get(player.id)
                    if bs is not None:
                        bs['deaths'] += 1
                    genome, fitness = self._bot_controller.unregister(player.id)
                    if genome is not None:
                        self._genome_pool.add(genome, fitness)
                        self._bot_death_count += 1
                        if self._bot_death_count % 50 == 0:
                            asyncio.create_task(self._genome_pool.save_async(config.NEAT_SAVE_PATH))
                            logger.info(
                                f"GA: gen={self._genome_pool.generation} "
                                f"deaths={self._genome_pool.total_deaths} "
                                f"pool_size={len(self._genome_pool._pool)}"
                            )
                    self._bot_ids.discard(player.id)
                    self.players.pop(player.id, None)
            else:
                await self._send_dead(player, score=0, killer_name="")
                self._respawn_player(player)

        # Food replenishment
        
        # Virus replenishment
        virus_deficit = config.VIRUS_TARGET - self.virus_mgr.count()
        if virus_deficit > 0:
            self.virus_mgr.respawn_to_target(config.VIRUS_TARGET)
        deficit = config.FOOD_TARGET - self.food_mgr.count()
        if deficit > 0:
            self.food_mgr.spawn_batch(deficit)

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
        # Get virus delta ONCE
        virus_new, virus_removed = self.virus_mgr.flush_delta()

        # ---- Send tick packets ----
        send_tasks = []
        for player in list(self.players.values()):
            if player.id in self._bot_ids:
                continue  # bots don't receive packets
            pkt = self._build_tick_packet(player, food_removed, virus_removed, leaderboard)
            send_tasks.append(self._safe_send(player.websocket, pkt))

        # ---- Bot burst spawning (NEAT only) ----
        if self._bot_mode != 'ppo':
            self._burst_elapsed += dt
            if self._burst_elapsed >= config.BOT_BURST_INTERVAL:
                self._burst_elapsed = 0.0

                # Cull bots below median fitness
                bot_players = [self.players[pid] for pid in list(self._bot_ids) if pid in self.players]
                if len(bot_players) >= 2:
                    fitnesses = sorted(self._bot_controller.current_fitness(p.id) for p in bot_players)
                    mid = len(fitnesses) // 2
                    median_fitness = (
                        (fitnesses[mid - 1] + fitnesses[mid]) / 2.0
                        if len(fitnesses) % 2 == 0
                        else fitnesses[mid]
                    )
                    culled = 0
                    for player in bot_players:
                        fitness = self._bot_controller.current_fitness(player.id)
                        if fitness < median_fitness:
                            genome, fitness = self._bot_controller.unregister(player.id)
                            if genome is not None:
                                self._genome_pool.add(genome, fitness)
                                self._bot_death_count += 1
                            # Clear cells so the player is removed from the world
                            for cell in list(player.cells):
                                self.cell_grid.remove(cell.id)
                                self.cell_map.pop(cell.id, None)
                            player.cells.clear()
                            self._bot_ids.discard(player.id)
                            self.players.pop(player.id, None)
                            culled += 1
                    if culled:
                        logger.info(
                            f"Bot cull: removed {culled} bots below median fitness {median_fitness:.1f}"
                        )

                # Bot respawn disabled: do not fill slots after cull
        # ---- Send to spectators ----
        if self._spectators:
            stats_pkt = None
            if self.tick_counter % config.LEADERBOARD_INTERVAL == 0:
                stats_pkt = self._build_stats_packet()
            for spec in list(self._spectators.values()):
                tick_pkt = self._build_spectator_tick_packet(spec, food_removed, virus_removed, leaderboard)
                send_tasks.append(self._safe_send(spec['ws'], tick_pkt))
                if stats_pkt is not None:
                    send_tasks.append(self._safe_send(spec['ws'], stats_pkt))

        if send_tasks:
            await asyncio.gather(*send_tasks, return_exceptions=True)
        else:
            await asyncio.sleep(0)  # yield to event loop when no packets to send

    # ------------------------------------------------------------------
    # Packet building
    # ------------------------------------------------------------------

    def _build_tick_packet(
        self,
        player: Player,
        food_removed_all: list[int],
        virus_removed_all: list[int],
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
        
        # Send all viruses in viewport that this client hasn't seen yet
        virus_new_visible = []
        virus_viewport_ids = self.virus_grid.query_rect(vx, vy, vw, vh)
        for vid in virus_viewport_ids:
            if vid not in player.sent_virus_ids:
                v = self.virus_mgr.get(vid)
                if v is not None:
                    player.sent_virus_ids.add(vid)
                    virus_new_visible.append(v)

        own_cell_ids = [c.id for c in player.cells]

        return protocol.encode_tick(
            tick_num=self.tick_counter,
            own_cell_ids=own_cell_ids,
            visible_cells=visible_cells,
            food_new=food_new_visible,
            food_removed=food_removed_all,  # all clients must remove these
            virus_new=virus_new_visible,
            virus_removed=virus_removed_all,
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
        virus_removed_all: list[int],
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
            # Virus delta for this viewport
            virus_new_visible = []
            for vid in self.virus_grid.query_rect(vx, vy, vw, vh):
                if vid not in spec.get('sent_virus_ids', set()):
                    v = self.virus_mgr.get(vid)
                    if v is not None:
                        if 'sent_virus_ids' not in spec:
                            spec['sent_virus_ids'] = set()
                        spec['sent_virus_ids'].add(vid)
                        virus_new_visible.append(v)
        else:
            # Overview: show entire world, skip food (too much data at once)
            vx, vy = 0.0, 0.0
            vw, vh = float(config.WORLD_W), float(config.WORLD_H)
            own_cell_ids = []
            food_new_visible = []
            virus_new_visible = []

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
            virus_new=virus_new_visible,
            virus_removed=virus_removed_all,
            known_player_ids=spec['known_player_ids'],
            player_map=self.players,
            leaderboard=leaderboard,
        )

    def _build_stats_packet(self) -> bytes:
        """Build a full player-stats snapshot for spectators."""
        players_info = []
        for pid, p in self.players.items():
            cx, cy = p.centroid
            is_bot = 1 if pid in self._bot_ids else 0
            fitness = round(self._bot_controller.current_fitness(pid), 1) if is_bot else 0.0
            players_info.append([
                pid,
                p.name,
                round(p.total_mass, 1),
                round(cx),
                round(cy),
                len(p.cells),
                is_bot,
                fitness,  # [7]
            ])
        return protocol.encode_stats(self.tick_counter, players_info, self.food_mgr.count())

    # ------------------------------------------------------------------
    # Player lifecycle
    # ------------------------------------------------------------------

    def _spawn_bot(self, genome=None) -> None:
        """Create a new bot player."""
        pid  = next(self._player_id_counter)
        name = _random_bot_name()
        bot  = Player(id=pid, name=name, websocket=NullWebSocket())
        self.players[pid] = bot
        self._bot_ids.add(pid)
        x = random.uniform(config.WORLD_W * 0.05, config.WORLD_W * 0.95)
        y = random.uniform(config.WORLD_H * 0.05, config.WORLD_H * 0.95)
        self._create_cell(bot, x, y, mass=10.0)
        if self._bot_mode == 'ppo':
            self._bot_controller.register(pid, x, y)
        else:
            if genome is None:
                genome = self._genome_pool.breed()
            bot.hue = genome_hue(genome)
            self._bot_controller.register(pid, x, y, genome)

    def _do_ppo_update(self) -> dict:
        """Run PPO update synchronously (called via asyncio.to_thread)."""
        return self._bot_controller.finish_rollout(self)

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
