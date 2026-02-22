"""Grid representation for the snake game."""

from __future__ import annotations

import enum

import numpy as np


class WallMode(enum.Enum):
    """Defines behavior when a snake reaches the grid boundary."""

    DEATH = "death"
    WRAP = "wrap"


class CellType(enum.IntEnum):
    """Integer codes stored in the grid array."""

    EMPTY = 0
    SNAKE = 1
    APPLE = 2


class Grid:
    """NumPy-backed game grid with configurable dimensions and wall mode.

    The grid stores cell states as integers for O(1) collision checks.
    Coordinates use (row, col) ordering consistent with NumPy indexing.
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        wall_mode: WallMode = WallMode.DEATH,
    ) -> None:
        if width < 4 or height < 4:
            raise ValueError("Grid dimensions must be at least 4×4.")
        self.width = width
        self.height = height
        self.wall_mode = wall_mode
        self.cells = np.zeros((height, width), dtype=np.int8)

    def clear(self) -> None:
        """Reset all cells to empty."""
        self.cells[:] = CellType.EMPTY

    def in_bounds(self, row: int, col: int) -> bool:
        """Check whether a coordinate lies within the grid."""
        return 0 <= row < self.height and 0 <= col < self.width

    def wrap(self, row: int, col: int) -> tuple[int, int]:
        """Wrap coordinates around the grid edges."""
        return row % self.height, col % self.width

    def get(self, row: int, col: int) -> CellType:
        """Return the cell type at the given coordinate."""
        return CellType(self.cells[row, col])

    def set(self, row: int, col: int, cell_type: CellType) -> None:
        """Set the cell type at the given coordinate."""
        self.cells[row, col] = cell_type

    def empty_cells(self) -> list[tuple[int, int]]:
        """Return a list of all empty cell coordinates."""
        rows, cols = np.where(self.cells == CellType.EMPTY)
        return list(zip(rows.tolist(), cols.tolist(), strict=True))

    def to_dict(self) -> dict:
        """Serialize grid state to a dictionary."""
        return {
            "width": self.width,
            "height": self.height,
            "wall_mode": self.wall_mode.value,
            "cells": self.cells.tolist(),
        }
