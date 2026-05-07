"""
NEAT-based genome pool for evolving bot neural networks.

Each genome is a neat.DefaultGenome whose weights/topology are evolved by
NEAT operators (crossover + mutation).  The network takes 97 sensory inputs
and produces 4 action outputs — no hand-coded thresholds or decision trees.

Input layout  (97 features):
  per sector: [food_proximity, food_mass_norm,
               prey_proximity, prey_smallness,
               threat_proximity, threat_danger]

Output layout (4):
  [move_x, move_y, split_signal, eject_signal]  — tanh ∈ [-1, 1]
"""
from __future__ import annotations

import itertools
import os
import pickle
import random
from pathlib import Path

import neat
from neat.innovation import InnovationTracker


# ---------------------------------------------------------------------------
# NEAT config (loaded once at module import)
# ---------------------------------------------------------------------------

_CFG_PATH = os.path.join(os.path.dirname(__file__), 'neat.cfg')
neat_config = neat.Config(
    neat.DefaultGenome,
    neat.DefaultReproduction,
    neat.DefaultSpeciesSet,
    neat.DefaultStagnation,
    _CFG_PATH,
)

# Shared innovation tracker — must be attached to genome_config before any
# genome is created or mutated.
_innovation_tracker = InnovationTracker()
neat_config.genome_config.innovation_tracker = _innovation_tracker


def random_genome() -> neat.DefaultGenome:
    """Create a new randomly-initialised NEAT genome."""
    g = neat.DefaultGenome(0)
    g.configure_new(neat_config.genome_config)
    return g


def genome_hue(genome: neat.DefaultGenome) -> int:
    """
    Derive a stable hue (0–359) from a genome's connection weights.

    NEAT crossover averages/inherits weights from both parents, so bred siblings
    have a mean weight close to their parents' — visually similar colours.
    Mutation drifts the hue slowly over generations.
    """
    weights = [cg.weight for cg in genome.connections.values() if cg.enabled]
    if not weights:
        # No connections yet (fresh genome) — use genome key as fallback
        return int(genome.key * 137.508) % 360
    mean_w = sum(weights) / len(weights)   # roughly in [-3, 3] for NEAT defaults
    # Map mean weight to [0, 360): clamp to [-3, 3] then scale
    t = max(-3.0, min(3.0, mean_w))        # t ∈ [-3, 3]
    hue = int((t + 3.0) / 6.0 * 360) % 360
    return hue


# ---------------------------------------------------------------------------
# Genome pool
# ---------------------------------------------------------------------------

_POOL_MAX_SIZE = 200
_TOURNAMENT_K  = 3


class GenomePool:
    """
    Maintains a population of NEAT genomes with their fitness scores.
    Supports tournament selection + NEAT crossover/mutation to breed new genomes.
    """

    def __init__(self) -> None:
        # List of {'genome': neat.DefaultGenome, 'fitness': float, 'generation': int}
        self._pool: list[dict] = []
        self._key_counter: int = 1
        self.generation: int = 0
        self.total_deaths: int = 0
        # Reset generation-level innovation deduplication on each new breeding round
        _innovation_tracker.reset_generation()

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------

    def _next_key(self) -> int:
        k = self._key_counter
        self._key_counter += 1
        return k

    def add(self, genome: neat.DefaultGenome, fitness: float) -> None:
        """Record a genome and its observed fitness."""
        self._pool.append({
            'genome':     genome,
            'fitness':    fitness,
            'generation': self.generation,
        })
        if len(self._pool) > _POOL_MAX_SIZE:
            self._pool.sort(key=lambda e: e['fitness'], reverse=True)
            self._pool = self._pool[:_POOL_MAX_SIZE]
        self.total_deaths += 1

    def breed(self) -> neat.DefaultGenome:
        """
        Produce a new child genome via tournament selection + NEAT crossover + mutation.
        Falls back to a fresh random genome if the pool is too small.
        """
        self.generation += 1
        _innovation_tracker.reset_generation()
        if len(self._pool) < 2:
            g = neat.DefaultGenome(self._next_key())
            g.configure_new(neat_config.genome_config)
            return g

        e1 = self._tournament_select()
        e2 = self._tournament_select()
        parent1, parent2 = e1['genome'], e2['genome']
        # neat.DefaultGenome.configure_crossover() uses .fitness to determine
        # which parent is "dominant" (higher fitness → more genes inherited).
        parent1.fitness = e1['fitness']
        parent2.fitness = e2['fitness']

        child = neat.DefaultGenome(self._next_key())
        child.configure_crossover(parent1, parent2, neat_config.genome_config)
        # Advance node_indexer past all node IDs inherited via crossover.
        # Without this, mutate_add_node can generate an ID that already exists.
        if child.nodes:
            max_node = max(child.nodes.keys())
            cfg = neat_config.genome_config
            if cfg.node_indexer is None:
                cfg.node_indexer = itertools.count(max_node + 1)
            else:
                # Drain until the counter is past max_node
                cur = next(cfg.node_indexer)
                if cur <= max_node:
                    cfg.node_indexer = itertools.count(max_node + 1)
                else:
                    cfg.node_indexer = itertools.count(cur)
        child.mutate(neat_config.genome_config)
        return child

    def _tournament_select(self) -> dict:
        k = min(_TOURNAMENT_K, len(self._pool))
        contestants = random.sample(self._pool, k)
        return max(contestants, key=lambda e: e['fitness'])

    def best(self, n: int = 5) -> list[tuple[float, neat.DefaultGenome]]:
        """Return the top-n (fitness, genome) pairs."""
        sorted_pool = sorted(self._pool, key=lambda e: e['fitness'], reverse=True)
        return [(e['fitness'], e['genome']) for e in sorted_pool[:n]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save pool to a pickle file. Atomic write via temp file."""
        data = {
            'generation':          self.generation,
            'total_deaths':        self.total_deaths,
            'key_counter':         self._key_counter,
            'innovation_counter':  _innovation_tracker.global_counter,
            'pool':                self._pool,
        }
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(data, f)
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> 'GenomePool':
        """Load pool from a pickle file. Returns an empty pool if file doesn't exist or is corrupt."""
        pool = GenomePool()
        p = Path(path)
        if not p.exists():
            return pool
        try:
            with open(p, 'rb') as f:
                data = pickle.load(f)
            pool.generation   = int(data.get('generation', 0))
            pool.total_deaths = int(data.get('total_deaths', 0))
            pool._key_counter = int(data.get('key_counter', 1))
            pool._pool        = data.get('pool', [])
            # Restore innovation tracker counter so IDs stay monotonically increasing
            saved_inno = data.get('innovation_counter', 0)
            if saved_inno > _innovation_tracker.global_counter:
                _innovation_tracker.global_counter = saved_inno
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load genome pool: {exc}")
        return pool
