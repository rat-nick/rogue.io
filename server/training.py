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
from .genetics import GenomePool, random_genome
from .food import FoodManager
from .player import Cell, Player
from .spatial import SpatialGrid
from .virus import VirusManager

logger = logging.getLogger(__name__)

GENERATION_TIME  = 120   # seconds per generation
TRAIN_POP_SIZE   = 200    # bots per generation
SURVIVE_FRACTION = 0.3   # top fraction whose genomes seed next generation
DIVERSITY_INJECTION_RATE = 0.15  # fraction of survivor slots replaced by fresh random genomes
TRAIN_PORT       = 8766   # separate port from main game
EARLY_NEXT_GEN_THRESHOLD = 0.5   # trigger early end when alive / start_pop <= this

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
        self._gen_start_pop:   int   = 0    # population at generation start

        # Early-next-gen: end the generation once alive bots fall below threshold
        self.early_next_gen:   bool  = True

        # Simulation time scale: >1 runs faster, <1 slower
        self.time_scale:       float = 1.0
        # Set to True by handle_viewer when time_scale changes so tick_loop resets its timer
        self._time_scale_changed: bool = False

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

    async def start_generation(self) -> None:
        """Wipe the world and spawn a fresh population for a new generation."""
        self._gen_records.clear()

        for pid in list(self._bot_ids):
            self._bot_controller.unregister(pid)
        self._bot_ids.clear()

        # Fully reset all spatial grids and entity state
        self.cell_grid  = SpatialGrid()
        self.food_grid  = SpatialGrid()
        self.virus_grid = SpatialGrid()
        self.players.clear()
        self.cell_map.clear()
        self.food_mgr  = FoodManager(self.food_grid)
        self.virus_mgr = VirusManager(self.virus_grid)
        self._cell_id_counter = itertools.count(1)

        # Reset per-viewer name caches (all players are new)
        for spec in self._viewers.values():
            spec['known_player_ids'].clear()

        self.seed_food()
        self.seed_viruses()

        self.generation       += 1
        self._gen_start_time   = asyncio.get_event_loop().time()
        self._gen_deaths       = 0

        # Spawn bots in batches, yielding periodically so the event loop stays alive
        for i in range(TRAIN_POP_SIZE):
            self._spawn_bot()
            if i % 20 == 0:
                await asyncio.sleep(0)

        self._gen_start_pop = len(self._bot_ids)

        logger.info(
            f"[Train] Generation {self.generation} started — {TRAIN_POP_SIZE} bots"
        )

    async def _end_generation(self) -> None:
        """Collect final fitness, select survivors, inject diversity, rebuild pool, save."""
        # Collect fitness for still-alive bots
        for pid in list(self._bot_ids):
            genome  = self._bot_controller.get_genome(pid)
            fitness = self._bot_controller.current_fitness(pid)
            if genome is not None:
                prev_fitness = self._gen_records.get(pid, (None, 0.0))[1]
                if fitness > prev_fitness:
                    self._gen_records[pid] = (genome, fitness)

        # Sort by raw fitness (highest first) — raw fitness determines who survives
        scored = sorted(
            self._gen_records.values(),
            key=lambda gf: gf[1],
            reverse=True,
        )
        n_survivors = max(1, int(len(scored) * SURVIVE_FRACTION))
        # Reserve a fraction of survivor slots for fresh random genomes so the
        # pool never fully converges to one cluster of similar high-fitness genomes.
        n_inject   = max(0, int(n_survivors * DIVERSITY_INJECTION_RATE))
        survivors  = scored[:n_survivors - n_inject]

        top_fit = survivors[0][1] if survivors else 0.0
        logger.info(
            f"[Train] Generation {self.generation} ended — "
            f"{len(survivors)}/{len(scored)} survivors + {n_inject} random injections, "
            f"top fitness={top_fit:.1f}"
        )

        # Rebuild pool: top survivors + fresh random genomes
        new_pool = [
            {'genome': g, 'fitness': f, 'generation': self.generation}
            for g, f in survivors
        ]
        for _ in range(n_inject):
            new_pool.append({
                'genome':     random_genome(),
                'fitness':    0.0,
                'generation': self.generation,
            })
        self._genome_pool._pool      = new_pool
        self._genome_pool.generation = self.generation

        # O(n²) species clustering — run in a thread so the event loop stays alive
        await asyncio.to_thread(self._genome_pool.update_species)
        logger.info(
            f"[Train] Species after selection: {self._genome_pool.species_count}"
        )
        # Non-blocking save — fire and forget so the tick loop isn't stalled
        asyncio.create_task(self._genome_pool.save_async(config.NEAT_SAVE_PATH))

    # ------------------------------------------------------------------
    # Tick loop
    # ------------------------------------------------------------------

    async def tick_loop(self) -> None:
        """Main training loop — runs indefinitely."""
        await self.start_generation()

        loop = asyncio.get_event_loop()
        next_tick = loop.time()

        while True:
            # If time_scale changed, push an immediate stats packet so clients see the new value
            if self._time_scale_changed:
                self._time_scale_changed = False
                if self._viewers:
                    stats_pkt = self._build_training_stats_packet()
                    await asyncio.gather(
                        *[self._safe_send(spec['ws'], stats_pkt) for spec in self._viewers.values()],
                        return_exceptions=True,
                    )
            next_tick += config.TICK_INTERVAL
            await self._tick(config.TICK_INTERVAL * self.time_scale)
            now = loop.time()
            sleep_time = next_tick - now
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                next_tick = loop.time()
                await asyncio.sleep(0)

    async def _tick(self, dt: float) -> None:
        self.tick_counter += 1

        # ---------- Bot AI ----------
        self._bot_controller.update(self, dt)

        # ---------- Physics ----------
        for player in list(self.players.values()):
            physics.apply_input(player, dt)

        physics.apply_split_velocity(self.players, self.cell_grid, dt)
        physics.update_collision_restore_ticks(self.players)

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

        # ---------- Corner instant-death ----------
        # Bots whose centroid is within VIRUS_RADIUS of any corner die immediately
        # and their genome is discarded (no offspring).
        corner_r = config.VIRUS_RADIUS
        for player in list(self.players.values()):
            if player.id not in self._bot_ids or not player.cells:
                continue
            cx, cy = player.centroid
            in_corner = (
                (cx < corner_r or cx > config.WORLD_W - corner_r) and
                (cy < corner_r or cy > config.WORLD_H - corner_r)
            )
            if in_corner:
                bs = self._bot_controller._state.get(player.id)
                if bs is not None:
                    bs['corner_death'] = True
                for cell in list(player.cells):
                    self.cell_grid.remove(cell.id)
                    self.cell_map.pop(cell.id, None)
                player.cells.clear()

        # ---------- Respawn dead bots ----------
        dead_bots = [
            p for p in list(self.players.values())
            if p.id in self._bot_ids and not p.cells
        ]
        for player in dead_bots:
            bs = self._bot_controller._state.get(player.id)
            is_corner_death = bs.get('corner_death', False) if bs else False
            if bs is not None:
                bs['deaths'] += 1
            genome, fitness = self._bot_controller.unregister(player.id)
            self._gen_deaths += 1
            if genome is not None and not is_corner_death:
                prev_fitness = self._gen_records.get(player.id, (None, 0.0))[1]
                if fitness > prev_fitness:
                    self._gen_records[player.id] = (genome, fitness)
            self._bot_ids.discard(player.id)
            self.players.pop(player.id, None)
            self._spawn_bot()
            await asyncio.sleep(0)

        # ---------- Early next-gen check ----------
        if (self.early_next_gen
                and self._gen_start_pop > 0
                and self._bot_ids
                and len(self._bot_ids) / self._gen_start_pop <= EARLY_NEXT_GEN_THRESHOLD):
            logger.info(
                f"[Train] Early next gen — {len(self._bot_ids)}/{self._gen_start_pop} bots alive"
            )
            self._gen_start_time = 0.0  # force generation boundary below

        # ---------- Food replenishment ----------
        deficit = config.FOOD_TARGET - self.food_mgr.count()
        if deficit > 0:
            self.food_mgr.spawn_batch(deficit)
        
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
            all_cells_now = [cell for cell, _ in self.cell_map.values()]

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
        if asyncio.get_event_loop().time() - self._gen_start_time >= GENERATION_TIME / self.time_scale:
            await self._end_generation()
            await self.start_generation()
            # Re-sync all connected viewers: their food/virus/cell state is stale
            # because start_generation() recreates managers with fresh IDs and
            # flushes the delta, so viewers never receive the new seeded food.
            # Sending MSG_INIT clears client state; the bootstrap tick restores it.
            if self._viewers:
                resync_tasks = []
                for vid, spec in self._viewers.items():
                    init_pkt      = protocol.encode_init(vid, config.WORLD_W, config.WORLD_H, config.TICK_RATE)
                    bootstrap_pkt = self._build_bootstrap_tick(spec['known_player_ids'])
                    resync_tasks.append(self._safe_send(spec['ws'], init_pkt))
                    resync_tasks.append(self._safe_send(spec['ws'], bootstrap_pkt))
                await asyncio.gather(*resync_tasks, return_exceptions=True)
        else:
            await asyncio.sleep(0)

    # ------------------------------------------------------------------
    # Packet building
    # ------------------------------------------------------------------

    def _build_bootstrap_tick(self, known_player_ids: set) -> bytes:
        """
        Build a tick packet treating all current world entities as newly spawned.
        Sent once to a viewer on connect so they see viruses/food/cells that
        already existed before they joined.
        """
        all_cells = [
            self.cell_map[cid][0]
            for cid in list(self.cell_grid._entity_key)
            if cid in self.cell_map
        ]
        all_food    = list(self.food_mgr._food.values())
        all_viruses = list(self.virus_mgr._viruses.values())
        return protocol.encode_tick(
            tick_num         = self.tick_counter,
            own_cell_ids     = [],
            visible_cells    = all_cells,
            food_new         = all_food,
            food_removed     = [],
            virus_new        = all_viruses,
            virus_removed    = [],
            known_player_ids = known_player_ids,
            player_map       = self.players,
            leaderboard      = None,
        )

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
            fitness = round(self._bot_controller.current_fitness(pid), 1)
            players_info.append([
                pid,
                p.name,
                round(p.total_mass, 1),
                round(cx),
                round(cy),
                len(p.cells),
                1,       # all are bots
                fitness, # [7]
            ])

        return protocol.encode_training_stats(
            tick_num       = self.tick_counter,
            generation     = self.generation,
            time_remaining = max(0.0, GENERATION_TIME / self.time_scale - (asyncio.get_event_loop().time() - self._gen_start_time)),
            pop_size       = len(self._bot_ids),
            top_fitness    = top_fitness,
            avg_fitness    = avg_fitness,
            best_mass      = best_mass,
            avg_mass       = avg_mass,
            total_deaths   = self._gen_deaths,
            players_info   = players_info,
            total_food     = self.food_mgr.count(),
            early_next_gen = self.early_next_gen,
            time_scale     = self.time_scale,
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
                # Bootstrap: send the full current world state so newly-connected viewers
                # immediately see all viruses, food, and cells that already exist.
                # Without this, viewers only receive delta updates and miss everything
                # spawned before they connected (viruses are seeded once at gen start).
                bootstrap_pkt = self._build_bootstrap_tick(spec['known_player_ids'])
                await websocket.send(bootstrap_pkt)
                await websocket.send(self._build_training_stats_packet())
            async for raw in websocket:
                # Accept manual next-gen trigger from client
                if isinstance(raw, bytes):
                    try:
                        msg = msgpack.unpackb(raw)
                    except Exception:
                        continue
                    if isinstance(msg, list) and msg:
                        if msg[0] == protocol.MSG_NEXT_GEN:
                            logger.info("[Train] Manual next generation triggered by client!")
                            self._gen_start_time = 0.0
                        elif msg[0] == protocol.MSG_SET_EARLY_NEXT_GEN:
                            self.early_next_gen = bool(msg[1]) if len(msg) > 1 else True
                            logger.info(f"[Train] Early next gen: {self.early_next_gen}")
                        elif msg[0] == protocol.MSG_SET_TIME_SCALE:
                            scale = float(msg[1]) if len(msg) > 1 else 1.0
                            self.time_scale = max(0.25, min(16.0, scale))
                            self._time_scale_changed = True
                            logger.info(f"[Train] Time scale set to {self.time_scale}x")
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
