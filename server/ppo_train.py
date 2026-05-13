"""PPO training world — a standalone training mode that uses a shared PPO
actor-critic policy instead of NEAT.

Run with:
    python -m server.ppo_train

Architecture:
  - All bots share one ActorCritic policy (weights updated via PPO).
  - On each game tick, observations are collected, actions sampled, and
    (obs, action, logprob, reward, done, value) stored in a RolloutBuffer.
  - After N_STEPS ticks the buffer is full: GAE advantages are computed and
    PPO mini-batch updates are run, then the rollout resets.
  - Dead bots are immediately respawned so the population stays constant.
  - A viewer WebSocket (TRAIN_PPO_PORT) broadcasts live game state +
    training stats (same protocol as the NEAT training viewer).
  - Model checkpoints saved to PPO_SAVE_PATH every N_STEPS rollouts.

Key hyperparameters (all configurable at runtime via viewer):
  N_BOTS         = 64     — bots in parallel (lower than NEAT pop for GPU memory)
  N_STEPS        = 128    — ticks per rollout per bot  (6.4 s of sim time)
  BATCH_SIZE     = 8192   — N_BOTS × N_STEPS
  MINIBATCH_SIZE = 1024
  N_EPOCHS       = 4
  LR             = 3e-4
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import math
import os
import random
from pathlib import Path

import msgpack
import numpy as np
import torch

from . import config
from . import physics
from . import protocol
from .bot import NullWebSocket
from .food import FoodManager
from .player import Cell, Player
from .spatial import SpatialGrid
from .virus import VirusManager
from .ppo_agent import ActorCritic, RolloutBuffer, PPOTrainer, N_OBS
from .ppo_bot import PPOBotController

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

N_BOTS         = 64
N_STEPS        = 128     # ticks per rollout (6.4 s at 20 Hz)
MINIBATCH_SIZE = 1024
N_EPOCHS       = 4
LR             = 3e-4
GAMMA          = 0.99
GAE_LAMBDA     = 0.95
CLIP_COEF      = 0.2
VALUE_COEF     = 0.5
ENTROPY_COEF   = 0.01
MAX_GRAD_NORM  = 0.5

TRAIN_PPO_PORT = 8767
PPO_SAVE_PATH  = os.path.join(os.path.dirname(__file__), "ppo_checkpoint.pt")

# Message type (matches app.js / train.js MSG_PPO_STATS = 0x30)
MSG_PPO_STATS = 0x30

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


# ---------------------------------------------------------------------------
# Training world
# ---------------------------------------------------------------------------

class PPOTrainingWorld:
    """Bot-only training world driven by a shared PPO policy."""

    # Viewer broadcast rates
    _BROADCAST_HZ     = 15.0
    _STATS_HZ         = 1.0
    _YIELD_INTERVAL_S = 0.020

    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)

        # Game world state
        self._player_id_counter = itertools.count(1)
        self._cell_id_counter   = itertools.count(1)
        self.players:  dict[int, Player]              = {}
        self.cell_map: dict[int, tuple[Cell, Player]] = {}
        self.cell_grid  = SpatialGrid()
        self.food_grid  = SpatialGrid()
        self.virus_grid = SpatialGrid()
        self.food_mgr   = FoodManager(self.food_grid)
        self.virus_mgr  = VirusManager(self.virus_grid)
        self.tick_counter: int = 0

        # PPO components
        self.model   = ActorCritic(n_obs=N_OBS).to(self.device)
        if Path(PPO_SAVE_PATH).exists():
            try:
                self.model = ActorCritic.load(PPO_SAVE_PATH, device=self.device)
                logger.info(f"[PPO] Loaded checkpoint from {PPO_SAVE_PATH}")
            except Exception as e:
                logger.warning(f"[PPO] Failed to load checkpoint: {e} — starting fresh")

        self.trainer = PPOTrainer(
            model=self.model,
            lr=LR,
            n_epochs=N_EPOCHS,
            minibatch_size=MINIBATCH_SIZE,
            clip_coef=CLIP_COEF,
            value_coef=VALUE_COEF,
            entropy_coef=ENTROPY_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            device=self.device,
        )
        self.buffer = RolloutBuffer(
            n_steps=N_STEPS,
            n_bots=N_BOTS,
            n_obs=N_OBS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            device=self.device,
        )
        self.bot_ctrl = PPOBotController(
            n_bots=N_BOTS,
            model=self.model,
            trainer=self.trainer,
            buffer=self.buffer,
            device=self.device,
        )

        # Tracking
        self._bot_ids: set[int] = set()
        self.n_rollouts:     int   = 0
        self.total_env_steps: int  = 0
        self._last_stats:    dict  = {}

        # Viewer deltas
        self._pending_food_new:      list = []
        self._pending_food_removed:  list = []
        self._pending_virus_new:     list = []
        self._pending_virus_removed: list = []
        self._viewers: dict[int, dict] = {}
        self._viewer_id_counter = itertools.count(1)

    # ------------------------------------------------------------------
    # World setup
    # ------------------------------------------------------------------

    def seed_food(self) -> None:
        self.food_mgr.spawn_batch(config.FOOD_TARGET)
        self.food_mgr.flush_delta()

    def seed_viruses(self) -> None:
        self.virus_mgr.respawn_to_target(config.VIRUS_TARGET)
        self.virus_mgr.flush_delta()

    def _spawn_bot(self) -> int:
        pid  = next(self._player_id_counter)
        name = _next_name()
        sx   = random.uniform(200.0, config.WORLD_W - 200.0)
        sy   = random.uniform(200.0, config.WORLD_H - 200.0)

        player = Player(id=pid, name=name, websocket=NullWebSocket())
        cell   = Cell(
            id=next(self._cell_id_counter),
            player_id=pid,
            x=sx, y=sy,
            mass=20.0,
        )
        player.cells.append(cell)
        player.target_x = sx
        player.target_y = sy
        self.players[pid]          = player
        self.cell_map[cell.id]     = (cell, player)
        self.cell_grid.insert(cell.id, cell.x, cell.y)

        self.bot_ctrl.register(pid, sx, sy)
        self._bot_ids.add(pid)
        return pid

    def _respawn_bot(self, player_id: int) -> None:
        """Respawn a dead bot (no genome needed — all bots share one policy)."""
        self.bot_ctrl.unregister(player_id)
        self._bot_ids.discard(player_id)
        self.players.pop(player_id, None)
        self._spawn_bot()

    # ------------------------------------------------------------------
    # Synchronous tick
    # ------------------------------------------------------------------

    def step(self, dt: float) -> None:
        """One synchronous simulation tick (no I/O)."""
        self.tick_counter += 1

        # ---- Bot AI ----
        self.bot_ctrl.update(self, dt)

        # ---- Physics ----
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
            self.cell_grid, self.cell_map, self.food_mgr, self._cell_id_counter,
        )
        for cell_id in virus_split_ids:
            self.cell_map.pop(cell_id, None)

        eaten_ids = physics.check_cell_collisions(self.players, self.cell_grid, self.cell_map)
        for cell_id in eaten_ids:
            self.cell_map.pop(cell_id, None)

        for player in list(self.players.values()):
            physics.perform_split(player, self.cell_grid, self.cell_map, self._cell_id_counter)
            ejected_ids = physics.perform_eject(player, self.food_mgr, self.food_grid)
            if ejected_ids:
                physics.check_ejected_virus_feeding(
                    ejected_ids, self.food_mgr, self.virus_mgr, self.virus_grid
                )

        # ---- Corner instant-death ----
        self._handle_corner_deaths()

        # ---- Handle dead bots (respawn immediately) ----
        dead = [p for p in self.players.values() if not p.cells and p.id in self._bot_ids]
        for player in dead:
            self._respawn_bot(player.id)

        # ---- Replenish food / viruses ----
        deficit = config.FOOD_TARGET - self.food_mgr.count()
        if deficit > 0:
            self.food_mgr.spawn_batch(deficit)
        virus_deficit = config.VIRUS_TARGET - self.virus_mgr.count()
        if virus_deficit > 0:
            self.virus_mgr.respawn_to_target(config.VIRUS_TARGET)

        # ---- Accumulate viewer deltas ----
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
        ckr_sq = config.CORNER_KILL_RADIUS ** 2
        corners = [
            (0.0, 0.0),
            (float(config.WORLD_W), 0.0),
            (0.0, float(config.WORLD_H)),
            (float(config.WORLD_W), float(config.WORLD_H)),
        ]
        for player in list(self.players.values()):
            doomed = [
                cell for cell in player.cells
                if any(
                    (cell.x - cx) ** 2 + (cell.y - cy) ** 2 <= ckr_sq
                    for cx, cy in corners
                )
            ]
            for cell in doomed:
                player.cells.remove(cell)
                self.cell_grid.remove(cell.id)
                self.cell_map.pop(cell.id, None)

    # ------------------------------------------------------------------
    # Async tick loop
    # ------------------------------------------------------------------

    async def tick_loop(self) -> None:
        """Sim runs sync and uncapped; viewer broadcasts are throttled."""
        # Initial world setup
        self.seed_food()
        self.seed_viruses()
        for _ in range(N_BOTS):
            self._spawn_bot()
            if _ % 16 == 0:
                await asyncio.sleep(0)
        logger.info(f"[PPO] Spawned {N_BOTS} bots — starting training")

        loop           = asyncio.get_event_loop()
        bcast_interval = 1.0 / self._BROADCAST_HZ
        stats_interval = 1.0 / self._STATS_HZ
        yield_interval = self._YIELD_INTERVAL_S
        last_bcast_t   = loop.time()
        last_stats_t   = loop.time()
        last_yield_t   = loop.time()
        sim_dt         = config.TICK_INTERVAL

        while True:
            self.step(sim_dt)

            # ---- PPO rollout update ----
            if self.bot_ctrl.is_rollout_full():
                stats = await asyncio.to_thread(self._do_ppo_update)
                self._last_stats = stats
                self.n_rollouts      += 1
                self.total_env_steps += N_STEPS * N_BOTS
                logger.info(
                    f"[PPO] Rollout {self.n_rollouts} | "
                    f"steps={self.total_env_steps} | "
                    f"pg={stats.get('pg_loss', 0):.4f} "
                    f"vf={stats.get('vf_loss', 0):.4f} "
                    f"ent={stats.get('entropy', 0):.4f} "
                    f"kl={stats.get('approx_kl', 0):.5f}"
                )
                if self.n_rollouts % 10 == 0:
                    self.model.save(PPO_SAVE_PATH)
                    logger.info(f"[PPO] Checkpoint saved to {PPO_SAVE_PATH}")
                self.bot_ctrl.reset_rollout()

            now = loop.time()
            if self._viewers and (now - last_bcast_t) >= bcast_interval:
                last_bcast_t = now
                send_stats   = (now - last_stats_t) >= stats_interval
                if send_stats:
                    last_stats_t = now
                await self._broadcast_to_viewers(send_stats=send_stats)

            if (now - last_yield_t) >= yield_interval:
                last_yield_t = now
                await asyncio.sleep(0)

    def _do_ppo_update(self) -> dict:
        """Run the PPO update synchronously (called via asyncio.to_thread)."""
        # Build last_values for bootstrapping from current observations
        stats = self.bot_ctrl.finish_rollout(self)
        return stats

    # ------------------------------------------------------------------
    # Viewer WebSocket handler
    # ------------------------------------------------------------------

    async def handle_viewer(self, websocket) -> None:
        vid = next(self._viewer_id_counter)
        self._viewers[vid] = {'ws': websocket, 'known_player_ids': set()}
        logger.info(f"[PPO] Viewer {vid} connected")
        try:
            await self._send_viewer_init(vid)
            async for raw in websocket:
                # Viewers may send param-update messages (msgpack dict)
                if isinstance(raw, bytes):
                    try:
                        msg = msgpack.unpackb(raw, raw=False)
                        if isinstance(msg, dict):
                            self._apply_viewer_params(msg)
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug(f"[PPO] Viewer {vid} error: {exc}")
        finally:
            self._viewers.pop(vid, None)
            logger.info(f"[PPO] Viewer {vid} disconnected")

    async def _send_viewer_init(self, vid: int) -> None:
        spec = self._viewers.get(vid)
        if spec is None:
            return
        ws    = spec['ws']
        known = spec['known_player_ids']
        init_pkt = protocol.encode_init(vid, config.WORLD_W, config.WORLD_H, config.TICK_RATE)
        await ws.send(init_pkt)
        # Bootstrap tick — send all existing cells/food/viruses so the viewer
        # doesn't have to wait for the next delta cycle to see anything.
        all_cells   = [cell for cell, _ in self.cell_map.values()]
        all_food    = list(self.food_mgr._food.values())
        all_viruses = list(self.virus_mgr._viruses.values())
        players_list = list(self.players.values())
        leaderboard  = sorted(
            [(p.name, p.score) for p in players_list], key=lambda x: -x[1]
        )[:10]
        bootstrap_pkt = protocol.encode_tick(
            tick_num         = self.tick_counter,
            own_cell_ids     = [],
            visible_cells    = all_cells,
            food_new         = all_food,
            food_removed     = [],
            virus_new        = all_viruses,
            virus_removed    = [],
            known_player_ids = known,
            player_map       = self.players,
            leaderboard      = leaderboard,
        )
        await ws.send(bootstrap_pkt)

    async def _broadcast_to_viewers(self, send_stats: bool = False) -> None:
        if not self._viewers:
            self._pending_food_new.clear()
            self._pending_food_removed.clear()
            self._pending_virus_new.clear()
            self._pending_virus_removed.clear()
            return

        food_new      = list(self._pending_food_new)
        food_removed  = list(self._pending_food_removed)
        virus_new     = list(self._pending_virus_new)
        virus_removed = list(self._pending_virus_removed)
        self._pending_food_new.clear()
        self._pending_food_removed.clear()
        self._pending_virus_new.clear()
        self._pending_virus_removed.clear()

        all_cells   = [cell for cell, _ in self.cell_map.values()]
        players_list = list(self.players.values())
        leaderboard  = sorted(
            [(p.name, p.score) for p in players_list], key=lambda x: -x[1]
        )[:10]

        stats_pkt = self._encode_ppo_stats() if send_stats else None

        dead_viewers = []
        for vid, spec in self._viewers.items():
            ws    = spec['ws']
            known = spec['known_player_ids']
            try:
                tick_pkt = protocol.encode_tick(
                    tick_num         = self.tick_counter,
                    own_cell_ids     = [],
                    visible_cells    = all_cells,
                    food_new         = food_new,
                    food_removed     = food_removed,
                    virus_new        = virus_new,
                    virus_removed    = virus_removed,
                    known_player_ids = known,
                    player_map       = self.players,
                    leaderboard      = leaderboard,
                )
                await ws.send(tick_pkt)
                if stats_pkt:
                    await ws.send(stats_pkt)
            except Exception:
                dead_viewers.append(vid)
        for vid in dead_viewers:
            self._viewers.pop(vid, None)

    def _encode_ppo_stats(self) -> bytes:
        # Build players_info in the same format as NEAT training stats so
        # train.js can populate the bot list and follow-panel identically.
        # Layout: [id, name, mass, cx, cy, cell_count, is_bot]
        players_info = []
        total_fitness = 0.0
        n_bots = 0
        for player in self.players.values():
            if player.id not in self._bot_ids:
                continue
            cx, cy = player.centroid
            fit = self.bot_ctrl.current_fitness(player.id)
            total_fitness += fit
            n_bots += 1
            players_info.append([
                player.id,
                player.name,
                round(player.total_mass, 1),
                round(cx, 1),
                round(cy, 1),
                len(player.cells),
                1,  # is_bot flag
            ])

        avg_fitness = total_fitness / n_bots if n_bots else 0.0
        stats = self._last_stats

        # [MSG_PPO_STATS, rollout, env_steps, n_bots, avg_fitness,
        #  pg_loss, vf_loss, entropy, approx_kl, players_info, food_count]
        return msgpack.packb([
            MSG_PPO_STATS,
            self.n_rollouts,
            self.total_env_steps,
            n_bots,
            round(avg_fitness, 2),
            round(stats.get('pg_loss',   0.0), 5),
            round(stats.get('vf_loss',   0.0), 5),
            round(stats.get('entropy',   0.0), 5),
            round(stats.get('approx_kl', 0.0), 6),
            players_info,
            self.food_mgr.count(),
        ])

    def _apply_viewer_params(self, params: dict) -> None:
        """Apply live param overrides sent by a viewer."""
        if 'lr' in params:
            for pg in self.trainer.optimizer.param_groups:
                pg['lr'] = float(params['lr'])
        if 'clip_coef' in params:
            self.trainer.clip_coef = float(params['clip_coef'])
        if 'entropy_coef' in params:
            self.trainer.entropy_coef = float(params['entropy_coef'])
        if 'n_epochs' in params:
            self.trainer.n_epochs = max(1, int(params['n_epochs']))
        logger.info(f"[PPO] Viewer params applied: {params}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    import websockets.asyncio.server as ws_server  # type: ignore

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"[PPO] Using device: {device_str}")

    world = PPOTrainingWorld(device=device_str)

    async def tick_loop_with_logging():
        try:
            await world.tick_loop()
        except Exception:
            logger.exception("[PPO] Tick loop crashed")
            raise

    tick_task = asyncio.create_task(tick_loop_with_logging())

    server = await ws_server.serve(
        world.handle_viewer,
        config.HOST,
        TRAIN_PPO_PORT,
    )
    logger.info(f"[PPO] Training server listening on ws://{config.HOST}:{TRAIN_PPO_PORT}")

    try:
        await server.serve_forever()
    finally:
        tick_task.cancel()
        server.close()
        await server.wait_closed()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    )
    asyncio.run(main())
