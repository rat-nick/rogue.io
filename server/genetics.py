"""
Genetic algorithm for evolving bot behaviour.

Genome: 8 floats that parameterise the bot's decision-making.
GenomePool: population of (genome, fitness) pairs with selection / crossover / mutation.
Persistence: load/save as JSON.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Gene definitions
# ---------------------------------------------------------------------------

# Each gene: (min_value, max_value)
GENE_BOUNDS: dict[str, tuple[float, float]] = {
    'food_seek_radius':    (200.0,  4000.0),   # how far to scan for food
    'threat_flee_radius':  (200.0,  4000.0),   # how far to detect threats
    'prey_chase_radius':   (200.0,  4000.0),   # how far to detect chaseable prey
    'flee_mass_ratio':     (1.05,   5.0),      # flee if enemy_mass > own * ratio
    'chase_mass_ratio':    (0.1,    0.95),     # chase if prey_mass < own * ratio
    'split_mass_threshold':(40.0,   800.0),    # split only above this own mass
    'wander_interval':     (0.5,    10.0),     # seconds between wander target picks
    'split_cooldown':      (4.0,    60.0),     # minimum seconds between splits
}


@dataclass
class BotGenome:
    food_seek_radius:     float = 1000.0
    threat_flee_radius:   float = 800.0
    prey_chase_radius:    float = 800.0
    flee_mass_ratio:      float = 1.3
    chase_mass_ratio:     float = 0.7
    split_mass_threshold: float = 150.0
    wander_interval:      float = 3.0
    split_cooldown:       float = 20.0

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> 'BotGenome':
        return BotGenome(**{k: float(v) for k, v in d.items() if k in GENE_BOUNDS})


def random_genome() -> BotGenome:
    """Uniformly random genome within gene bounds."""
    kwargs = {
        gene: random.uniform(lo, hi)
        for gene, (lo, hi) in GENE_BOUNDS.items()
    }
    return BotGenome(**kwargs)


def _clamp_genome(g: BotGenome) -> BotGenome:
    """Clamp all genes to their legal bounds."""
    d = g.to_dict()
    for gene, (lo, hi) in GENE_BOUNDS.items():
        d[gene] = max(lo, min(hi, d[gene]))
    return BotGenome.from_dict(d)


def crossover(a: BotGenome, b: BotGenome) -> BotGenome:
    """Uniform crossover: each gene is independently taken from a or b."""
    da, db = a.to_dict(), b.to_dict()
    child = {gene: (da[gene] if random.random() < 0.5 else db[gene])
             for gene in GENE_BOUNDS}
    return BotGenome.from_dict(child)


def mutate(genome: BotGenome, rate: float = 0.25, std_fraction: float = 0.12) -> BotGenome:
    """
    Gaussian mutation.
    Each gene is mutated with probability `rate`.
    The perturbation is Gaussian with std = std_fraction * gene_range.
    """
    d = genome.to_dict()
    for gene, (lo, hi) in GENE_BOUNDS.items():
        if random.random() < rate:
            std = (hi - lo) * std_fraction
            d[gene] = d[gene] + random.gauss(0.0, std)
    return _clamp_genome(BotGenome.from_dict(d))


# ---------------------------------------------------------------------------
# Genome pool
# ---------------------------------------------------------------------------

_POOL_MAX_SIZE = 200
_TOURNAMENT_K  = 3
_ELITE_COUNT   = 10


class GenomePool:
    """
    Maintains a population of genomes with their fitness scores.
    Supports tournament selection, crossover, and mutation to breed new genomes.
    """

    def __init__(self) -> None:
        # List of {'genome': BotGenome, 'fitness': float, 'generation': int}
        self._pool: list[dict] = []
        self.generation: int = 0
        self.total_deaths: int = 0

    # ------------------------------------------------------------------
    # Population management
    # ------------------------------------------------------------------

    def add(self, genome: BotGenome, fitness: float) -> None:
        """Record a genome and its observed fitness."""
        self._pool.append({
            'genome':     genome,
            'fitness':    fitness,
            'generation': self.generation,
        })
        # Keep pool bounded: always retain elites, trim the rest by fitness
        if len(self._pool) > _POOL_MAX_SIZE:
            self._pool.sort(key=lambda e: e['fitness'], reverse=True)
            self._pool = self._pool[:_POOL_MAX_SIZE]
        self.total_deaths += 1

    def breed(self) -> BotGenome:
        """
        Produce a new child genome.
        Uses tournament selection to pick two parents, then crossover + mutation.
        Falls back to a random genome if the pool is too small.
        """
        self.generation += 1
        if len(self._pool) < 2:
            return random_genome()

        parent_a = self._tournament_select()
        parent_b = self._tournament_select()
        child = crossover(parent_a, parent_b)
        child = mutate(child)
        return child

    def _tournament_select(self) -> BotGenome:
        k = min(_TOURNAMENT_K, len(self._pool))
        contestants = random.sample(self._pool, k)
        winner = max(contestants, key=lambda e: e['fitness'])
        return winner['genome']

    def best(self, n: int = 5) -> list[tuple[float, BotGenome]]:
        """Return the top-n (fitness, genome) pairs."""
        sorted_pool = sorted(self._pool, key=lambda e: e['fitness'], reverse=True)
        return [(e['fitness'], e['genome']) for e in sorted_pool[:n]]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save pool to JSON. Atomic write via temp file."""
        data = {
            'generation':   self.generation,
            'total_deaths': self.total_deaths,
            'pool': [
                {
                    'genome':     e['genome'].to_dict(),
                    'fitness':    e['fitness'],
                    'generation': e['generation'],
                }
                for e in self._pool
            ],
        }
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)

    @staticmethod
    def load(path: str) -> 'GenomePool':
        """Load pool from JSON. Returns an empty pool if file doesn't exist."""
        pool = GenomePool()
        p = Path(path)
        if not p.exists():
            return pool
        try:
            with open(p) as f:
                data = json.load(f)
            pool.generation   = int(data.get('generation', 0))
            pool.total_deaths = int(data.get('total_deaths', 0))
            for e in data.get('pool', []):
                genome = BotGenome.from_dict(e['genome'])
                pool._pool.append({
                    'genome':     genome,
                    'fitness':    float(e['fitness']),
                    'generation': int(e.get('generation', 0)),
                })
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(f"Failed to load genome pool: {exc}")
        return pool
