"""Tests for the AppleSpawner module."""

import numpy as np
import pytest

from smart_snake.apple import AppleSpawner
from smart_snake.grid import CellType, Grid


class TestAppleSpawnerInit:
    def test_default(self):
        grid = Grid(width=5, height=5)
        spawner = AppleSpawner(grid)
        assert spawner.max_apples == 1
        assert spawner.positions == []

    def test_invalid_max_apples(self):
        grid = Grid(width=5, height=5)
        with pytest.raises(ValueError, match="at least 1"):
            AppleSpawner(grid, max_apples=0)


class TestAppleSpawning:
    def test_spawn_one(self):
        grid = Grid(width=5, height=5)
        rng = np.random.default_rng(42)
        spawner = AppleSpawner(grid, max_apples=3, rng=rng)
        spawned = spawner.spawn(1)
        assert len(spawned) == 1
        r, c = spawned[0]
        assert grid.get(r, c) == CellType.APPLE

    def test_spawn_respects_max(self):
        grid = Grid(width=5, height=5)
        rng = np.random.default_rng(42)
        spawner = AppleSpawner(grid, max_apples=2, rng=rng)
        spawner.spawn(5)
        assert len(spawner.positions) == 2

    def test_spawn_deterministic(self):
        """Same seed produces same apple positions."""
        positions_a = self._spawn_with_seed(42)
        positions_b = self._spawn_with_seed(42)
        assert positions_a == positions_b

    def test_spawn_different_seeds(self):
        positions_a = self._spawn_with_seed(1)
        positions_b = self._spawn_with_seed(2)
        # Very unlikely to match with different seeds.
        assert positions_a != positions_b

    def test_spawn_on_full_grid(self):
        grid = Grid(width=4, height=4)
        grid.cells[:] = CellType.SNAKE
        spawner = AppleSpawner(grid, max_apples=1)
        assert spawner.spawn(1) == []

    @staticmethod
    def _spawn_with_seed(seed: int) -> list[tuple[int, int]]:
        grid = Grid(width=10, height=10)
        rng = np.random.default_rng(seed)
        spawner = AppleSpawner(grid, max_apples=3, rng=rng)
        return spawner.spawn(3)


class TestAppleRemoval:
    def test_remove_existing(self):
        grid = Grid(width=5, height=5)
        spawner = AppleSpawner(grid, max_apples=1)
        spawner.spawn(1)
        pos = spawner.positions[0]
        assert spawner.remove(pos[0], pos[1])
        assert grid.get(pos[0], pos[1]) == CellType.EMPTY
        assert len(spawner.positions) == 0

    def test_remove_nonexistent(self):
        grid = Grid(width=5, height=5)
        spawner = AppleSpawner(grid, max_apples=1)
        assert not spawner.remove(0, 0)


class TestAppleSerialization:
    def test_to_dict(self):
        grid = Grid(width=5, height=5)
        rng = np.random.default_rng(42)
        spawner = AppleSpawner(grid, max_apples=2, rng=rng)
        spawner.spawn(2)
        d = spawner.to_dict()
        assert d["max_apples"] == 2
        assert len(d["positions"]) == 2
