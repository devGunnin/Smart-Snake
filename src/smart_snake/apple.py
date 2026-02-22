"""Apple spawning logic."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from smart_snake.grid import Grid

logger = logging.getLogger(__name__)


class AppleSpawner:
    """Manages apple placement on the grid.

    Uses a seeded NumPy RNG for deterministic, reproducible placement.
    """

    def __init__(
        self,
        grid: Grid,
        max_apples: int = 1,
        rng: np.random.Generator | None = None,
    ) -> None:
        if max_apples < 1:
            raise ValueError("max_apples must be at least 1.")
        self.grid = grid
        self.max_apples = max_apples
        self.rng = rng if rng is not None else np.random.default_rng()
        self.positions: list[tuple[int, int]] = []

    def spawn(self, count: int = 1) -> list[tuple[int, int]]:
        """Spawn up to *count* apples on empty cells.

        Returns the list of newly spawned positions.
        """
        from smart_snake.grid import CellType

        empty = self.grid.empty_cells()
        if not empty:
            logger.warning("No empty cells available for apple spawning.")
            return []

        needed = min(count, self.max_apples - len(self.positions), len(empty))
        if needed <= 0:
            return []

        indices = self.rng.choice(len(empty), size=needed, replace=False)
        spawned: list[tuple[int, int]] = []
        for idx in indices:
            pos = empty[idx]
            self.grid.set(pos[0], pos[1], CellType.APPLE)
            self.positions.append(pos)
            spawned.append(pos)

        return spawned

    def remove(self, row: int, col: int) -> bool:
        """Remove an apple at the given position. Returns True if removed."""
        from smart_snake.grid import CellType

        pos = (row, col)
        if pos in self.positions:
            self.positions.remove(pos)
            self.grid.set(row, col, CellType.EMPTY)
            return True
        return False

    def to_dict(self) -> dict:
        """Serialize apple state to a dictionary."""
        return {
            "positions": [list(p) for p in self.positions],
            "max_apples": self.max_apples,
        }
