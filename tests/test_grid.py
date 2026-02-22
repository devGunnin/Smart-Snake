"""Tests for the Grid module."""

import numpy as np
import pytest

from smart_snake.grid import CellType, Grid, WallMode


class TestGridInit:
    def test_default_dimensions(self):
        grid = Grid()
        assert grid.width == 20
        assert grid.height == 20
        assert grid.wall_mode == WallMode.DEATH

    def test_custom_dimensions(self):
        grid = Grid(width=10, height=8)
        assert grid.width == 10
        assert grid.height == 8
        assert grid.cells.shape == (8, 10)

    def test_minimum_size_enforced(self):
        with pytest.raises(ValueError, match="at least 4"):
            Grid(width=3, height=4)
        with pytest.raises(ValueError, match="at least 4"):
            Grid(width=4, height=3)

    def test_all_cells_start_empty(self):
        grid = Grid(width=5, height=5)
        assert np.all(grid.cells == CellType.EMPTY)


class TestGridOperations:
    def test_set_and_get(self):
        grid = Grid(width=5, height=5)
        grid.set(2, 3, CellType.SNAKE)
        assert grid.get(2, 3) == CellType.SNAKE

    def test_clear(self):
        grid = Grid(width=5, height=5)
        grid.set(0, 0, CellType.SNAKE)
        grid.set(1, 1, CellType.APPLE)
        grid.clear()
        assert np.all(grid.cells == CellType.EMPTY)

    def test_in_bounds(self):
        grid = Grid(width=5, height=5)
        assert grid.in_bounds(0, 0)
        assert grid.in_bounds(4, 4)
        assert not grid.in_bounds(-1, 0)
        assert not grid.in_bounds(0, 5)
        assert not grid.in_bounds(5, 0)

    def test_wrap(self):
        grid = Grid(width=5, height=5)
        assert grid.wrap(-1, 0) == (4, 0)
        assert grid.wrap(0, -1) == (0, 4)
        assert grid.wrap(5, 5) == (0, 0)

    def test_empty_cells(self):
        grid = Grid(width=4, height=4)
        assert len(grid.empty_cells()) == 16
        grid.set(0, 0, CellType.SNAKE)
        grid.set(1, 1, CellType.APPLE)
        assert len(grid.empty_cells()) == 14


class TestGridSerialization:
    def test_to_dict_structure(self):
        grid = Grid(width=5, height=5, wall_mode=WallMode.WRAP)
        d = grid.to_dict()
        assert d["width"] == 5
        assert d["height"] == 5
        assert d["wall_mode"] == "wrap"
        assert len(d["cells"]) == 5
        assert len(d["cells"][0]) == 5

    def test_to_dict_reflects_state(self):
        grid = Grid(width=4, height=4)
        grid.set(1, 2, CellType.APPLE)
        d = grid.to_dict()
        assert d["cells"][1][2] == CellType.APPLE
