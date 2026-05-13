from __future__ import annotations
from . import config


class SpatialGrid:
    """
    Uniform grid spatial index.
    Two separate instances are used: one for cells, one for food.
    """

    def __init__(self):
        # (gx, gy) -> set of entity IDs
        self._grid: dict[tuple[int, int], set[int]] = {}
        # entity_id -> grid key (for O(1) removal/move)
        self._entity_key: dict[int, tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _key(x: float, y: float) -> tuple[int, int]:
        return (int(x / config.GRID_CELL_SIZE), int(y / config.GRID_CELL_SIZE))

    def _bucket(self, key: tuple[int, int]) -> set[int]:
        bucket = self._grid.get(key)
        if bucket is None:
            bucket = set()
            self._grid[key] = bucket
        return bucket

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, entity_id: int, x: float, y: float) -> None:
        key = self._key(x, y)
        self._bucket(key).add(entity_id)
        self._entity_key[entity_id] = key

    def remove(self, entity_id: int) -> None:
        key = self._entity_key.pop(entity_id, None)
        if key is not None:
            bucket = self._grid.get(key)
            if bucket is not None:
                bucket.discard(entity_id)
                if not bucket:
                    del self._grid[key]

    def move(self, entity_id: int, new_x: float, new_y: float) -> None:
        """Update entity position. Only touches the grid when grid key changes."""
        new_key = self._key(new_x, new_y)
        old_key = self._entity_key.get(entity_id)
        if old_key == new_key:
            return
        # Remove from old bucket
        if old_key is not None:
            old_bucket = self._grid.get(old_key)
            if old_bucket is not None:
                old_bucket.discard(entity_id)
                if not old_bucket:
                    del self._grid[old_key]
        # Insert into new bucket
        self._bucket(new_key).add(entity_id)
        self._entity_key[entity_id] = new_key

    def query_rect(self, x: float, y: float, w: float, h: float) -> list[int]:
        """Return all entity IDs in grid cells overlapping the given world rect.
        Returns a list, not a set — every entity lives in exactly one bucket so
        results are already disjoint, and list.extend is much cheaper than
        set.update for the common case of many small buckets."""
        cs = config.GRID_CELL_SIZE
        gx_min = int(x / cs)
        gx_max = int((x + w) / cs)
        gy_min = int(y / cs)
        gy_max = int((y + h) / cs)
        result: list[int] = []
        grid = self._grid
        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                bucket = grid.get((gx, gy))
                if bucket:
                    result.extend(bucket)
        return result

    def query_radius(self, x: float, y: float, r: float) -> list[int]:
        """Return entity IDs within radius r of (x, y). Rect query + distance filter."""
        return self.query_rect(x - r, y - r, r * 2, r * 2)

    def __len__(self) -> int:
        return len(self._entity_key)
