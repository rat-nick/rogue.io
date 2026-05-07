"""
Training-mode game world.

Runs a closed bot-only training session:
  - 100 bots spawned at the start of each generation (seeded from the genome pool)
  - Bots run for GENERATION_TIME seconds
  - On death, bots are immediately respawned with a fresh bred genome; fitness is recorded
  - At the end of each generation the top SURVIVE_FRACTION (by peak fitness) seed the next
    generation; pool is rebuilt from survivors only
  - Training viewer clients connect via WebSocket and receive:
      * MSG_TICK every tick  (standard format, full world, no viewport culling)
      * MSG_TRAINING_STATS every LEADERBOARD_INTERVAL ticks (training HUD data)

Entry point: python -m server.train
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import random

import msgpack
import neat

from . import config
from . import physics
from . import protocol
from .bot import BotController, NullWebSocket
from .genetics import GenomePool
from .food import FoodManager
from .player import Cell, Player
from .spatial import SpatialGrid
from .virus import VirusManager

logger = logging.getLogger(__name__)

GENERATION_TIME  = 60.0   # seconds per generation
TRAIN_POP_SIZE   = 100    # bots per generation
SURVIVE_FRACTION = 0.50   # top fraction whose genomes seed next generation
TRAIN_PORT       = 8766   # separate port from main game

_BOT_NAMES = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon",
    "Zeta", "Eta", "Theta", "Iota", "Kappa",
    "Lambda", "Mu", "Nu", "Xi", "Omicron",
    "Pi", "Rho", "Sigma", "Tau", "Upsilon",
    "Chi", "Psi", "Omega",
]
_name_idx = 0


def _next_name() -> str:
    global _name_idx
    name = _BOT_NAMES[_name_idx % len(_BOT_NAMES)]
    _name_idx += 1
    return name


class TrainingWorld:
    def __init__(self) -> None:
        self._player_id_counter = itertools.count(1)
        self._cell_id_counter   = itertools.count(1)

        self.players:  dict[int, Player]              = {}
        self.cell_map: dict[int, tuple[Cell, Player]] = {}

        self.cell_grid = SpatialGrid()
        self.food_grid = SpatialGrid()
        self.virus_grid = SpatialGrid()
        self.food_mgr  = FoodManager(self.food_grid)
        self.virus_mgr = VirusManager(self.virus_grid)

        self.tick_counter: int = 0

        self._bot_ids:        set[int]      = set()
        self._bot_controller: BotController = BotController()
        self._genome_pool:    GenomePool    = GenomePool.load(config.NEAT_SAVE_PATH)

        # Per-generation fitness records: pid -> (genome, fitness)
        self._gen_records: dict[int, tuple[neat.DefaultGenome, float]] = {}

        # Generation tracking
        self.generation:       int   = 0
        self._gen_start_time:  float = 0.0  # asyncio loop time at generation start
        self._gen_deaths:      int   = 0

        # Viewer connections: viewer_id -> {'ws': websocket, 'known_player_ids': set()}
        self._viewers: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def seed_food(self) -> None:
        self.food_mgr.spawn_batch(config.FOOD_TARGET)
        self.food_mgr.flush_delta()  # clear delta so seed food is treated as baseline
    
    def seed_viruses(self) -> None:
        self.virus_mgr.spawn_corners()
        self.virus_mgr.respawn_to_target(config.VIRUS_TARGET)
        self.virus_mgr.flush_delta()

    def start_generation(self) -> None:
        """Wipe the world and spawn a fresh population for a new generation."""
        self._gen_records.clear()

        for pid in list(self._bot_ids):
            self._bot_controller.unregister(pid)
        self._bot_ids.clear()
        self.virus_grid = SpatialGrid()
        self.food_mgr  = FoodManager(self.food_grid)
        self.virus_mgr = VirusManager(self.virus_grid)
        self._cell_id_counter = itertools.count(1)

        # Reset per-viewer name caches (all players are new)
        for spec in self._viewers.values():
            spec['known_player_ids'].clear()

        self.seed_food()
        self.seed_viruses
        # Reset per-viewer name caches (all players are new)
        for spec in self._viewers.values():
            spec['known_player_ids'].clear()

        self.seed_food()

        self.generation       += 1
        self._gen_start_time   = asyncio.get_event_loop().time()
        self._gen_deaths       = 0

        for _ in range(TRAIN_POP_SIZE):
            self._spawn_bot()

        logger.info(
            f"[Train] Generation {self.generation} started — {TRAIN_POP_SIZE} bots"
        )

    def _end_generation(self) -> None:
        """Collect final fitness, select survivors, rebuild pool, save."""
        # Collect fitness for still-alive bots
        for pid in list(self._bot_ids):
            genome  = self._bot_controller.get_genome(pid)
            fitness = self._bot_controller.current_fitness(pid)
            if genome is not None:
                prev_fitness = self._gen_records.get(pid, (None, 0.0))[1]
                if fitness > prev_fitness:
                    self._gen_records[pid] = (genome, fitness)

        # Sort by fitness (highest first)
        scored = sorted(
            self._gen_records.values(),
            key=lambda gf: gf[1],
            reverse=True,
        )
        n_survivors = max(1, int(len(scored) * SURVIVE_FRACTION))
        survivors   = scored[:n_survivors]

        top_fit = survivors[0][1] if survivors else 0.0
        logger.info(
            f"[Train] Generation {self.generation} ended — "
            f"{n_survivors}/{len(scored)} survivors, "
            f"top fitness={top_fit:.1f}"
        )

        # Rebuild pool with only the survivor genomes
        self._genome_pool._pool = [
            {'genome': g, 'fitness': f, 'generation': self.generation}
            for g, f in survivors
        ]
        self._genome_pool.generation = self.generation
        self._genome_pool.save(config.NEAT_SAVE_PATH)

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------

    async def tick_loop(self) -> None:
        """Main training loop — runs indefinitely."""
        self.start_generation()

        loop = asyncio.get_event_loop()
        next_tick = loop.time()

        while True:
            next_tick += config.TICK_INTERVAL
            await self._tick()
            now = loop.time()
            sleep_time = next_tick - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                next_tick = loop.time()
                await asyncio.sleep(0)

    async def _tick(self) -> None:
        dt = config.TICK_INTERVAL
        self.tick_counter += 1

        # ---------- Bot AI ----------
        self._bot_controller.update(self, dt)

        # ---------- Physics ----------
        for player in list(self.players.values()):
            physics.apply_input(player, dt)

        physics.apply_split_velocity(self.players, self.cell_grid, dt)

        all_cells = [c for p in self.players.values() for c in p.cells]
        physics.update_positions(all_cells, self.cell_grid)

        decayed = physics.apply_decay(self.players, dt)
        for _pid, cell_id in decayed:
            self.cell_grid.remove(cell_id)
            self.cell_map.pop(cell_id, None)

        # Check virus collisions
        virus_split_ids = physics.check_virus_collisions(
            self.players, self.virus_mgr, self.virus_grid,
            self.cell_grid, self.cell_map, self.food_mgr, self._cell_id_counter
        )
        for cell_id in virus_split_ids:
            self.cell_map.pop(cell_id, None)

        physics.apply_merge_attraction(self.players, self.cell_grid, dt)
        physics.update_merge_timers(
            self.players, self.cell_grid, self.food_mgr, self.cell_map, dt
        )
        self.food_mgr.tick_decay(dt)

        physics.check_food_collisions(self.players, self.food_mgr, self.food_grid)

        # Check virus collisions
        virus_split_ids = physics.check_virus_collisions(
            self.players, self.virus_mgr, self.virus_grid,
            self.cell_grid, self.cell_map, self.food_mgr, self._cell_id_counter
        )
        for cell_id in virus_split_ids:
            self.cell_map.pop(cell_id, None)

        eaten_ids = physics.check_cell_collisions(
            self.players, self.cell_grid, self.cell_map
        )
        for cell_id in eaten_ids:
            self.cell_map.pop(cell_id, None)

        for player in list(self.players.values()):
            physics.perform_split(
                player, self.cell_grid, self.cell_map, self._cell_id_counter
            )
            ejected_ids = physics.perform_eject(player, self.food_mgr, self.food_grid)
            # Check if ejected mass feeds viruses
            if ejected_ids:
                physics.check_ejected_virus_feeding(ejected_ids, self.food_mgr, self.virus_mgr, self.virus_grid)

        # ---------- Respawn dead bots ----------
        dead_bots = [
            p for p in list(self.players.values())
            if p.id in self._bot_ids and not p.cells
        ]
        for player in dead_bots:
            bs = self._bot_controller._state.get(player.id)
            if bs is not None:
                bs['deaths'] += 1
            genome, fitness = self._bot_controller.unregister(player.id)
            self._gen_deaths += 1
            if genome is not None:
                prev_fitness = self._gen_records.get(player.id, (None, 0.0))[1]
                if fitness > prev_fitness:
                    self._gen_records[player.id] = (genome, fitness)
                self._genome_pool.add(genome, fitness)
            self._bot_ids.discard(player.id)
            self.players.pop(player.id, None)
            # No respawn here; bots only respawn at generation start
            await asyncio.sleep(0)

        # ---------- Food replenishment ----------
        deficit = config.FOOD_TARGET - self.food_mgr.count()
        if deficit > 0:
            self.food_mgr.spawn_batch(min(deficit, 50))
        
        # ---------- Virus replenishment ----------
        virus_deficit = config.VIRUS_TARGET - self.virus_mgr.count()
        if virus_deficit > 0:
            self.virus_mgr.respawn_to_target(config.VIRUS_TARGET)

        # ---------- Flush food delta (once per tick, shared by all viewers) ----------
        food_new, food_removed = self.food_mgr.flush_delta()
        # Flush virus delta
        virus_new, virus_removed = self.virus_mgr.flush_delta()

        # ---------- Send to viewers ----------
        if self._viewers:
            # Build cell list (whole world, no culling)
            all_cells_now = [
                self.cell_map[cid][0]
                for cid in list(self.cell_grid._entity_key)
                if cid in self.cell_map
            ]

            # Leaderboard built from current mass rankings
            lb = None
            if self.tick_counter % config.LEADERBOARD_INTERVAL == 0:
                sorted_players = sorted(
                    self.players.values(),
                    key=lambda p: p.total_mass,
                    reverse=True,
                )[:10]
                lb = [[p.name, p.score] for p in sorted_players]

            # Per-viewer tick packets (name dedup is per-viewer)
            tick_tasks = []
            for spec in self._viewers.values():
                pkt = protocol.encode_tick(
                    tick_num         = self.tick_counter,
                    own_cell_ids     = [],
                    visible_cells    = all_cells_now,
                    food_new         = food_new,
                    food_removed     = food_removed,
                    virus_new        = virus_new,
                    virus_removed    = virus_removed,
                    known_player_ids = spec['known_player_ids'],
                    player_map       = self.players,
                    leaderboard      = lb,
                )
                tick_tasks.append(self._safe_send(spec['ws'], pkt))
            await asyncio.gather(*tick_tasks, return_exceptions=True)

            # Training stats every LEADERBOARD_INTERVAL ticks
            if self.tick_counter % config.LEADERBOARD_INTERVAL == 0:
                stats_pkt   = self._build_training_stats_packet()
                stats_tasks = [self._safe_send(spec['ws'], stats_pkt) for spec in self._viewers.values()]
                await asyncio.gather(*stats_tasks, return_exceptions=True)

        # ---------- Generation boundary ----------
        if asyncio.get_event_loop().time() - self._gen_start_time >= GENERATION_TIME:
            self._end_generation()
            self.start_generation()
        else:
            await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Packet building
    # ------------------------------------------------------------------

    def _build_training_stats_packet(self) -> bytes:
        fitnesses = [
            self._bot_controller.current_fitness(pid)
            for pid in self._bot_ids
            if pid in self.players
        ]
        top_fitness = max(fitnesses, default=0.0)
        avg_fitness = (sum(fitnesses) / len(fitnesses)) if fitnesses else 0.0

        masses    = [self.players[pid].total_mass for pid in self._bot_ids if pid in self.players]
        best_mass = max(masses, default=0.0)
        avg_mass  = (sum(masses) / len(masses)) if masses else 0.0

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
                1,  # all are bots
            ])

        return protocol.encode_training_stats(
            tick_num       = self.tick_counter,
            generation     = self.generation,
            time_remaining = max(0.0, GENERATION_TIME - (asyncio.get_event_loop().time() - self._gen_start_time)),
            pop_size       = len(self._bot_ids),
            top_fitness    = top_fitness,
            avg_fitness    = avg_fitness,
            best_mass      = best_mass,
            avg_mass       = avg_mass,
            total_deaths   = self._gen_deaths,
            players_info   = players_info,
            total_food     = self.food_mgr.count(),
        )

    # ------------------------------------------------------------------
    # Viewer WebSocket handler
    # ------------------------------------------------------------------

    async def handle_viewer(self, websocket) -> None:
        """WebSocket handler for training viewer clients."""
        vid = next(self._player_id_counter)
        spec = {'ws': websocket, 'known_player_ids': set()}
        self._viewers[vid] = spec
        logger.info(f"[Train] Viewer {vid} connected")
        try:
            init_pkt = protocol.encode_init(
                vid, config.WORLD_W, config.WORLD_H, config.TICK_RATE
            )
            await websocket.send(init_pkt)
            if self.generation > 0:
                await websocket.send(self._build_training_stats_packet())
            async for raw in websocket:
                # Accept manual next-gen trigger from client
                if isinstance(raw, bytes):
                    try:
                        msg = msgpack.unpackb(raw)
                    except Exception:
                        continue
                    if isinstance(msg, list) and msg and msg[0] == getattr(protocol, 'MSG_NEXT_GEN', 0x25):
                        logger.info("[Train] Manual next generation triggered by client!")
                        self._gen_start_time = 0.0  # force next gen
        except Exception as exc:
            logger.debug(f"[Train] Viewer {vid} error: {exc}")
        finally:
            self._viewers.pop(vid, None)
            logger.info(f"[Train] Viewer {vid} disconnected")

    # ------------------------------------------------------------------
    # Bot lifecycle helpers
    # ------------------------------------------------------------------

    def _spawn_bot(self) -> None:
        """Breed a genome from the pool and spawn a new bot."""
        genome = self._genome_pool.breed()
        pid    = next(self._player_id_counter)
        name   = _next_name()
        bot    = Player(id=pid, name=name, websocket=NullWebSocket())
        self.players[pid] = bot
        self._bot_ids.add(pid)
        x = random.uniform(config.WORLD_W * 0.05, config.WORLD_W * 0.95)
        y = random.uniform(config.WORLD_H * 0.05, config.WORLD_H * 0.95)
        self._create_cell(bot, x, y, mass=20.0)
        self._bot_controller.register(pid, x, y, genome)

    def _create_cell(self, player: Player, x: float, y: float, mass: float) -> Cell:
        cell_id = next(self._cell_id_counter)
        cell    = Cell(id=cell_id, player_id=player.id, x=x, y=y, mass=mass)
        player.cells.append(cell)
        self.cell_grid.insert(cell_id, x, y)
        self.cell_map[cell_id] = (cell, player)
        return cell

    @staticmethod
    async def _safe_send(websocket, data: bytes) -> None:
        try:
            await websocket.send(data)
        except Exception:
            pass
