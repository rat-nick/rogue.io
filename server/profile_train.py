"""Headless synchronous profiler for the training world.

Drives `TrainingWorld.step()` directly (no asyncio, no viewers) so cProfile
sees the real CPU cost of one tick. Also reports clean wall-clock TPS when
profiling is disabled.

Usage:
    python -m server.profile_train [warmup_ticks] [measure_ticks] [--no-profile]
"""
from __future__ import annotations

import asyncio
import cProfile
import pstats
import random
import sys
import time

from . import config
from .training import TrainingWorld


def _bootstrap(world: TrainingWorld) -> None:
    """Synchronous equivalent of `await world.start_generation()` for profiling.
    Same physics-relevant work, no asyncio yields."""
    world._gen_records.clear()
    for pid in list(world._bot_ids):
        world._bot_controller.unregister(pid)
    world._bot_ids.clear()
    world.cell_grid.__init__()
    world.food_grid.__init__()
    world.virus_grid.__init__()
    world.players.clear()
    world.cell_map.clear()
    from .food import FoodManager
    from .virus import VirusManager
    world.food_mgr = FoodManager(world.food_grid)
    world.virus_mgr = VirusManager(world.virus_grid)
    import itertools
    world._cell_id_counter = itertools.count(1)
    world.seed_food()
    world.seed_viruses()
    world.generation += 1
    world._gen_start_tick = world.tick_counter
    world._gen_deaths = 0
    world._bot_controller.reset_species()
    for _ in range(world.train_pop_size):
        world._spawn_bot()
    world._gen_start_pop = len(world._bot_ids)


def _run(warmup: int, measure: int, profile: bool) -> None:
    random.seed(42)
    world = TrainingWorld()
    _bootstrap(world)

    for _ in range(warmup):
        world.step(config.TICK_INTERVAL)

    pr = cProfile.Profile() if profile else None
    t0 = time.perf_counter()
    if pr is not None:
        pr.enable()
    for _ in range(measure):
        world.step(config.TICK_INTERVAL)
    if pr is not None:
        pr.disable()
    elapsed = time.perf_counter() - t0

    tps = measure / elapsed
    print()
    print(f"Ticks measured : {measure}")
    print(f"Wall time      : {elapsed:.3f} s")
    print(f"Sim TPS        : {tps:.1f}  ({'profiled' if profile else 'clean'})")
    print(f"Real-time x    : {tps / config.TICK_RATE:.1f}x  (TICK_RATE={config.TICK_RATE})")
    print(f"Bots alive     : {len(world._bot_ids)}")
    print(f"Cells alive    : {len(world.cell_map)}")
    print(f"Food count     : {world.food_mgr.count()}")
    print()

    if pr is not None:
        stats = pstats.Stats(pr).sort_stats('cumulative')
        print("---- top 25 by cumulative time ----")
        stats.print_stats(25)
        print("---- top 25 by tottime ----")
        stats.sort_stats('tottime').print_stats(25)


def main() -> None:
    profile = '--no-profile' not in sys.argv
    pos_args = [a for a in sys.argv[1:] if not a.startswith('--')]
    warmup  = int(pos_args[0]) if len(pos_args) > 0 else 100
    measure = int(pos_args[1]) if len(pos_args) > 1 else 500
    print(f"Warmup={warmup} ticks, measure={measure} ticks, profile={profile}")
    _run(warmup, measure, profile)


if __name__ == '__main__':
    main()
