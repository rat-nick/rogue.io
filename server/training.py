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
from .genetics import GenomePool, random_genome, neat_config
from .food import FoodManager
from .player import Cell, Player
from .spatial import SpatialGrid
from .virus import VirusManager

logger = logging.getLogger(__name__)

GENERATION_TIME  = 120  # seconds per generation — longer trace + recurrent topology
                        # need more ticks for sophisticated behavior to manifest
TRAIN_POP_SIZE   = 200    # bots per generation
SURVIVE_FRACTION = 0.3  # top fraction whose genomes seed next generation —
                         # tighter selection drives exploitation once the
                         # population starts diverging into competent strategies
DIVERSITY_INJECTION_RATE = 0.10  # fresh-random genomes carry no Hebbian gains
                                 # and rarely contribute; keep low but non-zero
TRAIN_PORT       = 8766   # separate port from main game
EARLY_NEXT_GEN_THRESHOLD = 0.4   # let more bots die before cutting short —
                                 # gives top performers more learning time per gen

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

        # Generation tracking — measured in sim ticks (not wall time).
        # Sim runs uncapped; one "sim second" == TICK_RATE ticks regardless of wall clock.
        self.generation:       int   = 0
        self._gen_start_tick:  int   = 0    # tick_counter at generation start
        self._gen_deaths:      int   = 0
        self._gen_start_pop:   int   = 0    # population at generation start
        self._force_next_gen:  bool  = False

        # Early-next-gen: end the generation once alive bots fall below threshold
        self.early_next_gen:   bool  = True

        # Time scale: multiplier applied to dt every tick. 2× = twice as much game
        # time per tick → physics, decay, and generation timer all advance 2× faster.
        self.time_scale:       float = 1.0

        # Per-broadcast accumulation buffers — sim flushes deltas every tick,
        # broadcaster (15 Hz) drains these into outbound packets.
        self._pending_food_new:      list = []
        self._pending_food_removed:  list = []
        self._pending_virus_new:     list = []
        self._pending_virus_removed: list = []
        self._gen_just_rolled:       bool = False  # broadcast loop sees this and resyncs viewers

        # Configurable training params — instance vars so they can be changed live
        self.generation_time:  float = float(GENERATION_TIME)
        self.train_pop_size:   int   = TRAIN_POP_SIZE
        self.survive_fraction: float = SURVIVE_FRACTION
        self.diversity_rate:   float = DIVERSITY_INJECTION_RATE
        self.early_threshold:  float = EARLY_NEXT_GEN_THRESHOLD
        # Params queued by the viewer, applied at the start of the next generation
        self._pending_params:  dict  = {}

        # Viewer connections: viewer_id -> {'ws': websocket, 'known_player_ids': set()}
        self._viewers: dict[int, dict] = {}

    # ------------------------------------------------------------------
    # Live param control
    # ------------------------------------------------------------------

    def _apply_pending_params(self) -> None:
        p = self._pending_params
        if 'generation_time'  in p:
            self.generation_time  = float(p['generation_time'])
        if 'pop_size'         in p:
            self.train_pop_size   = max(1, int(p['pop_size']))
        if 'survive_fraction' in p:
            self.survive_fraction = float(p['survive_fraction'])
        if 'diversity_rate'   in p:
            self.diversity_rate   = float(p['diversity_rate'])
        if 'early_threshold'  in p:
            self.early_threshold  = float(p['early_threshold'])
        if 'tournament_k'     in p:
            self._genome_pool.tournament_k = max(2, int(p['tournament_k']))
        self._bot_controller.apply_params(p)
        gc = neat_config.genome_config
        sc = neat_config.species_set_config
        if 'weight_mutate_power'  in p:
            gc.weight_mutate_power  = float(p['weight_mutate_power'])
        if 'weight_mutate_rate'   in p:
            gc.weight_mutate_rate   = float(p['weight_mutate_rate'])
        if 'weight_replace_rate'  in p:
            gc.weight_replace_rate  = float(p['weight_replace_rate'])
        if 'conn_add_prob'        in p:
            gc.conn_add_prob        = float(p['conn_add_prob'])
        if 'conn_delete_prob'     in p:
            gc.conn_delete_prob     = float(p['conn_delete_prob'])
        if 'node_add_prob'        in p:
            gc.node_add_prob        = float(p['node_add_prob'])
        if 'node_delete_prob'     in p:
            gc.node_delete_prob     = float(p['node_delete_prob'])
        if 'compat_threshold'     in p:
            sc.compatibility_threshold = float(p['compat_threshold'])
        logger.info(f"[Train] Applied params: {p}")

    def _current_params(self) -> dict:
        fw = self._bot_controller._fw
        return {
            'generation_time':  self.generation_time,
            'pop_size':         self.train_pop_size,
            'survive_fraction': self.survive_fraction,
            'diversity_rate':   self.diversity_rate,
            'early_threshold':  self.early_threshold,
            'tournament_k':     self._genome_pool.tournament_k,
            'fw_peak':          fw['peak'],
            'fw_avg':           fw['avg'],
            'fw_food':          fw['food'],
            'fw_cells':         fw['cells'],
            'fw_survival':      fw['survival'],
            'fw_corner':        fw['corner'],
            'fw_death_exp':     fw['death_exp'],
            'hebbian_lr':         self._bot_controller._hebbian_lr,
            'hebbian_decay':      self._bot_controller._hebbian_decay,
            'weight_mutate_power':neat_config.genome_config.weight_mutate_power,
            'weight_mutate_rate': neat_config.genome_config.weight_mutate_rate,
            'weight_replace_rate':neat_config.genome_config.weight_replace_rate,
            'conn_add_prob':      neat_config.genome_config.conn_add_prob,
            'conn_delete_prob':   neat_config.genome_config.conn_delete_prob,
            'node_add_prob':      neat_config.genome_config.node_add_prob,
            'node_delete_prob':   neat_config.genome_config.node_delete_prob,
            'compat_threshold':   neat_config.species_set_config.compatibility_threshold,
        }

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def seed_food(self) -> None:
        self.food_mgr.spawn_batch(config.FOOD_TARGET)
        self.food_mgr.flush_delta()  # clear delta so seed food is treated as baseline
    
    def seed_viruses(self) -> None:
        self.virus_mgr.respawn_to_target(config.VIRUS_TARGET)
        self.virus_mgr.flush_delta()

    async def start_generation(self) -> None:
        """Wipe the world and spawn a fresh population for a new generation."""
        if self._pending_params:
            self._apply_pending_params()
            self._pending_params.clear()
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
        # Drop pending viewer deltas — they reference stale IDs from the old gen.
        self._pending_food_new.clear()
        self._pending_food_removed.clear()
        self._pending_virus_new.clear()
        self._pending_virus_removed.clear()

        # Reset per-viewer name caches (all players are new)
        for spec in self._viewers.values():
            spec['known_player_ids'].clear()

        self.seed_food()
        self.seed_viruses()

        self.generation       += 1
        self._gen_start_tick   = self.tick_counter
        self._gen_deaths       = 0
        self._force_next_gen   = False

        # Reset species representatives so each generation gets a fresh assignment
        self._bot_controller.reset_species()

        # Spawn bots in batches, yielding periodically so the event loop stays alive
        for i in range(self.train_pop_size):
            self._spawn_bot()
            if i % 50 == 0:
                await asyncio.sleep(0)

        self._gen_start_pop = len(self._bot_ids)
        self._gen_just_rolled = True

        logger.info(
            f"[Train] Generation {self.generation} started — {self.train_pop_size} bots"
        )

    async def _end_generation(self) -> None:
        """Collect final fitness, select survivors, inject diversity, rebuild pool, save."""
        # Flush Hebbian weights from the numpy W matrix into genome objects so that
        # survivors' genomes carry learned weights into the pool.
        self._bot_controller.write_back_all()

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
        n_survivors = max(1, int(len(scored) * self.survive_fraction))
        # Reserve a fraction of survivor slots for fresh random genomes so the
        # pool never fully converges to one cluster of similar high-fitness genomes.
        n_inject   = max(0, int(n_survivors * self.diversity_rate))
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

    # ----- TICK LOOP -----
    # Sim runs uncapped: `step()` is fully synchronous and called back-to-back.
    # Viewer broadcasts are throttled to a fixed wall-clock rate via a separate
    # async helper, so they never gate sim throughput. Yielding to the asyncio
    # event loop happens once per N ticks so connection accept / viewer recv
    # tasks still get CPU. The previous design awaited `asyncio.to_thread` for
    # bot AI, which on Windows cost ~30 ms per IOCP roundtrip and dominated wall
    # time at 400 bots — that's what's gone now.

    _BROADCAST_HZ      = 15.0   # viewer game-state packet rate (cells/food/viruses)
    _STATS_HZ          = 1.0    # training stats packet rate (sidebar, bot list, charts)
    _YIELD_INTERVAL_S  = 0.020  # yield to asyncio at most every 20 ms

    async def tick_loop(self) -> None:
        """Main training loop. Sim ticks run sync and uncapped; viewer broadcasts
        and connection housekeeping are interleaved at fixed wall-clock rates."""
        await self.start_generation()

        loop = asyncio.get_event_loop()
        bcast_interval  = 1.0 / self._BROADCAST_HZ
        stats_interval  = 1.0 / self._STATS_HZ
        yield_interval  = self._YIELD_INTERVAL_S
        last_bcast_t    = loop.time()
        last_stats_t    = loop.time()
        last_yield_t    = loop.time()
        sim_dt = config.TICK_INTERVAL

        while True:
            self.step(sim_dt * self.time_scale)
            now = loop.time()

            # Generation boundary: each tick advances (sim_dt * time_scale) game-seconds
            elapsed_sim_seconds = (self.tick_counter - self._gen_start_tick) * sim_dt * self.time_scale
            if self._force_next_gen or elapsed_sim_seconds >= self.generation_time:
                await self._end_generation()
                await self.start_generation()
                if self._viewers:
                    await self._resync_viewers()
                now = loop.time()
                last_bcast_t = now
                last_stats_t = now
                last_yield_t = now
                continue

            if self._viewers and (now - last_bcast_t) >= bcast_interval:
                last_bcast_t = now
                send_stats = (now - last_stats_t) >= stats_interval
                if send_stats:
                    last_stats_t = now
                await self._broadcast_to_viewers(send_stats=send_stats)

            if (now - last_yield_t) >= yield_interval:
                last_yield_t = now
                await asyncio.sleep(0)

    def step(self, dt: float) -> None:
        """One synchronous simulation tick. No I/O, no awaits — runs as fast as the CPU allows."""
        self.tick_counter += 1

        # ---------- Bot AI ----------
        self._bot_controller.update(self, dt)

        # ---------- Physics ----------
        players = self.players
        cell_grid = self.cell_grid
        cell_map = self.cell_map

        for player in list(players.values()):
            physics.apply_input(player, dt)

        physics.apply_split_velocity(players, cell_grid, dt)
        physics.update_collision_restore_ticks(players)

        all_cells = [c for p in players.values() for c in p.cells]
        physics.update_positions(all_cells, cell_grid)

        decayed = physics.apply_decay(players, dt)
        for _pid, cell_id in decayed:
            cell_grid.remove(cell_id)
            cell_map.pop(cell_id, None)

        physics.apply_merge_attraction(players, cell_grid, dt)
        physics.update_merge_timers(players, cell_grid, cell_map, dt)
        self.food_mgr.tick_decay(dt)

        physics.check_food_collisions(players, self.food_mgr, self.food_grid)

        virus_split_ids = physics.check_virus_collisions(
            players, self.virus_mgr, self.virus_grid,
            cell_grid, cell_map, self.food_mgr, self._cell_id_counter,
        )
        for cell_id in virus_split_ids:
            cell_map.pop(cell_id, None)

        eaten_ids = physics.check_cell_collisions(players, cell_grid, cell_map)
        for cell_id in eaten_ids:
            cell_map.pop(cell_id, None)

        for player in list(players.values()):
            physics.perform_split(player, cell_grid, cell_map, self._cell_id_counter)
            ejected_ids = physics.perform_eject(player, self.food_mgr, self.food_grid)
            if ejected_ids:
                physics.check_ejected_virus_feeding(ejected_ids, self.food_mgr, self.virus_mgr, self.virus_grid)

        # ---------- Corner instant-death ----------
        self._handle_corner_deaths()

        # ---------- Respawn dead bots ----------
        self._handle_dead_bots()

        # ---------- Early next-gen check ----------
        if (self.early_next_gen
                and self._gen_start_pop > 0
                and self._bot_ids
                and len(self._bot_ids) / self._gen_start_pop <= self.early_threshold):
            logger.info(
                f"[Train] Early next gen — {len(self._bot_ids)}/{self._gen_start_pop} bots alive"
            )
            self._force_next_gen = True

        # ---------- Replenish ----------
        deficit = config.FOOD_TARGET - self.food_mgr.count()
        if deficit > 0:
            self.food_mgr.spawn_batch(deficit)
        virus_deficit = config.VIRUS_TARGET - self.virus_mgr.count()
        if virus_deficit > 0:
            self.virus_mgr.respawn_to_target(config.VIRUS_TARGET)

        # ---------- Accumulate viewer deltas ----------
        food_new, food_removed = self.food_mgr.flush_delta()
        if food_new:
            self._pending_food_new.extend(food_new)
        if food_removed:
            self._pending_food_removed.extend(food_removed)
        virus_new, virus_removed = self.virus_mgr.flush_delta()
        if virus_new:
            self._pending_virus_new.extend(virus_new)
        if virus_removed:
            self._pending_virus_removed.extend(virus_removed)

    def _handle_corner_deaths(self) -> None:
        """Bots whose centroid is within VIRUS_RADIUS of any corner die instantly
        and their genome is discarded (no offspring)."""
        corner_r = config.VIRUS_RADIUS
        world_w = config.WORLD_W
        world_h = config.WORLD_H
        cell_grid = self.cell_grid
        cell_map = self.cell_map
        bs_get = self._bot_controller._state.get
        for player in list(self.players.values()):
            if player.id not in self._bot_ids or not player.cells:
                continue
            cx, cy = player.centroid
            in_corner = (
                (cx < corner_r or cx > world_w - corner_r) and
                (cy < corner_r or cy > world_h - corner_r)
            )
            if in_corner:
                bs = bs_get(player.id)
                if bs is not None:
                    bs['corner_death'] = True
                for cell in list(player.cells):
                    cell_grid.remove(cell.id)
                    cell_map.pop(cell.id, None)
                player.cells.clear()

    def _handle_dead_bots(self) -> None:
        """Record fitness for any bot with no surviving cells, then respawn."""
        dead_bots = [
            p for p in list(self.players.values())
            if p.id in self._bot_ids and not p.cells
        ]
        if not dead_bots:
            return
        bs_state = self._bot_controller._state
        for player in dead_bots:
            bs = bs_state.get(player.id)
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

    async def _broadcast_to_viewers(self, send_stats: bool = True) -> None:
        """Send game state to all viewers; optionally include training stats.

        Game-state (cells/food/viruses) is sent every call at _BROADCAST_HZ.
        Stats (sidebar numbers, bot list) are sent only when send_stats=True,
        which happens at _STATS_HZ (1 Hz). Decoupling these rates prevents the
        full DOM bot-list rebuild from firing 15× per second and killing clicks.
        """
        if not self._viewers:
            self._pending_food_new.clear()
            self._pending_food_removed.clear()
            self._pending_virus_new.clear()
            self._pending_virus_removed.clear()
            return

        food_new = self._pending_food_new
        food_removed = self._pending_food_removed
        virus_new = self._pending_virus_new
        virus_removed = self._pending_virus_removed
        self._pending_food_new = []
        self._pending_food_removed = []
        self._pending_virus_new = []
        self._pending_virus_removed = []

        all_cells_now = [cell for cell, _ in self.cell_map.values()]

        sorted_players = sorted(
            self.players.values(), key=lambda p: p.total_mass, reverse=True,
        )[:10]
        lb = [[p.name, p.score] for p in sorted_players]

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

        if send_stats:
            stats_pkt = self._build_training_stats_packet()
            for spec in self._viewers.values():
                tick_tasks.append(self._safe_send(spec['ws'], stats_pkt))

        await asyncio.gather(*tick_tasks, return_exceptions=True)

    async def _resync_viewers(self) -> None:
        """Send MSG_INIT + bootstrap tick + fresh stats after a generation rollover."""
        stats_pkt = self._build_training_stats_packet()
        tasks = []
        for vid, spec in self._viewers.items():
            init_pkt = protocol.encode_init(vid, config.WORLD_W, config.WORLD_H, config.TICK_RATE)
            bootstrap_pkt = self._build_bootstrap_tick(spec['known_player_ids'])
            tasks.append(self._safe_send(spec['ws'], init_pkt))
            tasks.append(self._safe_send(spec['ws'], bootstrap_pkt))
            tasks.append(self._safe_send(spec['ws'], stats_pkt))
        await asyncio.gather(*tasks, return_exceptions=True)

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
            bs = self._bot_controller._state.get(pid)
            species_id = bs.get('species_id', -1) if bs else -1
            players_info.append([
                pid,
                p.name,
                round(p.total_mass, 1),
                round(cx),
                round(cy),
                len(p.cells),
                1,          # all are bots
                fitness,    # [7]
                species_id, # [8]
            ])

        sim_seconds_elapsed = (self.tick_counter - self._gen_start_tick) * config.TICK_INTERVAL * self.time_scale
        time_remaining = max(0.0, self.generation_time - sim_seconds_elapsed)
        return protocol.encode_training_stats(
            tick_num       = self.tick_counter,
            generation     = self.generation,
            time_remaining = time_remaining,
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
            params         = self._current_params(),
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
                            self._force_next_gen = True
                        elif msg[0] == protocol.MSG_SET_EARLY_NEXT_GEN:
                            self.early_next_gen = bool(msg[1]) if len(msg) > 1 else True
                            logger.info(f"[Train] Early next gen: {self.early_next_gen}")
                        elif msg[0] == protocol.MSG_SET_TIME_SCALE:
                            scale = float(msg[1]) if len(msg) > 1 else 1.0
                            self.time_scale = max(0.25, min(16.0, scale))
                            logger.info(f"[Train] Time scale set to {self.time_scale}x")
                        elif msg[0] == protocol.MSG_SET_TRAINING_PARAMS:
                            if len(msg) > 1 and isinstance(msg[1], dict):
                                self._pending_params.update(msg[1])
                                logger.info(f"[Train] Params queued for next gen: {msg[1]}")
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
