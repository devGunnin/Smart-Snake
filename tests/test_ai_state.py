"""Tests for state encoding."""

import numpy as np
import pytest

from smart_snake.ai.state import (
    CH_APPLES,
    CH_DIR_DOWN,
    CH_DIR_LEFT,
    CH_DIR_RIGHT,
    CH_DIR_UP,
    CH_NEAREST_APPLE_DISTANCE,
    CH_NEAREST_DANGER_DISTANCE,
    CH_OPPONENT_BODIES,
    CH_OPPONENT_HEADS,
    CH_OPPONENT_MEAN_LENGTH,
    CH_OWN_BODY,
    CH_OWN_HEAD,
    CH_OWN_LENGTH,
    CH_WALLS,
    NUM_CHANNELS,
    encode_multi,
    encode_single,
)
from smart_snake.engine import GameEngine
from smart_snake.grid import CellType, Grid, WallMode
from smart_snake.multiplayer import MatchConfig, MultiplayerEngine
from smart_snake.snake import Direction, Snake


class TestEncodeSingle:
    def test_shape(self):
        engine = GameEngine(width=10, height=10, seed=0)
        obs = encode_single(engine.grid, engine.snake)
        assert obs.shape == (NUM_CHANNELS, 10, 10)
        assert obs.dtype == np.float32

    def test_own_head_channel(self):
        engine = GameEngine(width=10, height=10, seed=0)
        obs = encode_single(engine.grid, engine.snake)
        hr, hc = engine.snake.head
        assert obs[CH_OWN_HEAD, hr, hc] == 1.0
        assert obs[CH_OWN_HEAD].sum() == 1.0

    def test_own_body_channel(self):
        engine = GameEngine(width=10, height=10, seed=0)
        obs = encode_single(engine.grid, engine.snake)
        assert obs[CH_OWN_BODY].sum() == len(engine.snake.body)

    def test_opponent_channels_empty_in_single(self):
        engine = GameEngine(width=10, height=10, seed=0)
        obs = encode_single(engine.grid, engine.snake)
        assert obs[CH_OPPONENT_HEADS].sum() == 0.0
        assert obs[CH_OPPONENT_BODIES].sum() == 0.0

    def test_apple_channel(self):
        engine = GameEngine(width=10, height=10, max_apples=3, seed=0)
        obs = encode_single(engine.grid, engine.snake)
        apple_count = (engine.grid.cells == CellType.APPLE).sum()
        assert obs[CH_APPLES].sum() == apple_count

    def test_wall_channel_with_obstacles(self):
        grid = Grid(width=10, height=10)
        grid.set(0, 0, CellType.OBSTACLE)
        grid.set(1, 1, CellType.OBSTACLE)
        snake = Snake(5, 5, Direction.RIGHT, length=3)
        obs = encode_single(grid, snake)
        assert obs[CH_WALLS, 0, 0] == 1.0
        assert obs[CH_WALLS, 1, 1] == 1.0
        # Death-mode boundaries are also encoded as walls (plus interior obstacle).
        assert obs[CH_WALLS].sum() == 37.0

    def test_wall_channel_includes_death_mode_boundaries(self):
        grid = Grid(width=10, height=10, wall_mode=WallMode.DEATH)
        snake = Snake(5, 5, Direction.RIGHT, length=3)
        obs = encode_single(grid, snake)

        assert obs[CH_WALLS, 0, 5] == 1.0
        assert obs[CH_WALLS, 9, 5] == 1.0
        assert obs[CH_WALLS, 5, 0] == 1.0
        assert obs[CH_WALLS, 5, 9] == 1.0
        assert obs[CH_WALLS, 5, 5] == 0.0
        assert obs[CH_WALLS].sum() == 36.0

    def test_wall_channel_excludes_wrap_mode_boundaries(self):
        grid = Grid(width=10, height=10, wall_mode=WallMode.WRAP)
        snake = Snake(5, 5, Direction.RIGHT, length=3)
        obs = encode_single(grid, snake)
        assert obs[CH_WALLS].sum() == 0.0

    def test_dead_snake_empty_channels(self):
        grid = Grid(width=10, height=10)
        snake = Snake(5, 5, Direction.RIGHT, length=3)
        snake.alive = False
        obs = encode_single(grid, snake)
        assert obs[CH_OWN_HEAD].sum() == 0.0
        assert obs[CH_OWN_BODY].sum() == 0.0

    def test_auxiliary_channels(self):
        engine = GameEngine(width=10, height=10, max_apples=3, seed=0)
        obs = encode_single(engine.grid, engine.snake)

        assert 0.0 < obs[CH_OWN_LENGTH, 0, 0] < 1.0
        assert obs[CH_OPPONENT_MEAN_LENGTH, 0, 0] == 0.0

        assert obs[CH_DIR_RIGHT].sum() == 100.0
        assert obs[CH_DIR_UP].sum() == 0.0
        assert obs[CH_DIR_DOWN].sum() == 0.0
        assert obs[CH_DIR_LEFT].sum() == 0.0

        assert 0.0 <= obs[CH_NEAREST_APPLE_DISTANCE, 0, 0] <= 1.0
        assert 0.0 <= obs[CH_NEAREST_DANGER_DISTANCE, 0, 0] <= 1.0

    def test_nearest_danger_excludes_unreachable_neck_segment(self):
        grid = Grid(width=10, height=10, wall_mode=WallMode.DEATH)
        snake = Snake(5, 5, Direction.RIGHT, length=2)
        obs = encode_single(grid, snake)

        # With only head + neck, nearest danger should come from map boundaries.
        expected = 4.0 / 18.0
        assert obs[CH_NEAREST_DANGER_DISTANCE, 0, 0] == pytest.approx(expected)

    def test_nearest_danger_ignores_boundaries_in_wrap_mode(self):
        grid = Grid(width=10, height=10, wall_mode=WallMode.WRAP)
        snake = Snake(5, 5, Direction.RIGHT, length=2)
        obs = encode_single(grid, snake)
        assert obs[CH_NEAREST_DANGER_DISTANCE, 0, 0] == 1.0

    def test_relative_mode_centers_head(self):
        grid = Grid(width=10, height=10)
        snake = Snake(2, 7, Direction.RIGHT, length=3)

        abs_obs = encode_single(grid, snake, mode="absolute")
        rel_obs = encode_single(grid, snake, mode="relative")

        assert abs_obs[CH_OWN_HEAD, 2, 7] == 1.0
        assert rel_obs[CH_OWN_HEAD, 5, 5] == 1.0

    def test_invalid_mode_raises(self):
        engine = GameEngine(width=10, height=10, seed=0)
        with pytest.raises(ValueError, match="Unsupported state encoding mode"):
            encode_single(engine.grid, engine.snake, mode="bad-mode")


class TestEncodeMulti:
    def test_shape(self):
        engine = MultiplayerEngine(MatchConfig(player_count=2, seed=0))
        obs = encode_multi(engine.grid, engine.snakes, 0)
        h, w = engine.config.effective_height, engine.config.effective_width
        assert obs.shape == (NUM_CHANNELS, h, w)

    def test_perspective_own_head(self):
        engine = MultiplayerEngine(MatchConfig(player_count=2, seed=0))
        obs0 = encode_multi(engine.grid, engine.snakes, 0)
        obs1 = encode_multi(engine.grid, engine.snakes, 1)
        # Snake 0's head should be in own-head channel of obs0.
        h0r, h0c = engine.snakes[0].head
        assert obs0[CH_OWN_HEAD, h0r, h0c] == 1.0
        # Snake 0's head should be in opponent-head channel of obs1.
        assert obs1[CH_OPPONENT_HEADS, h0r, h0c] == 1.0

    def test_opponent_body_channel(self):
        engine = MultiplayerEngine(MatchConfig(player_count=2, seed=0))
        obs0 = encode_multi(engine.grid, engine.snakes, 0)
        # Snake 1's body should be in opponent-bodies channel.
        for r, c in engine.snakes[1].body:
            assert obs0[CH_OPPONENT_BODIES, r, c] == 1.0

    def test_three_players(self):
        engine = MultiplayerEngine(MatchConfig(player_count=3, seed=0))
        obs0 = encode_multi(engine.grid, engine.snakes, 0)
        # Both snake 1 and 2 heads should be in opponent-heads channel.
        h1r, h1c = engine.snakes[1].head
        h2r, h2c = engine.snakes[2].head
        assert obs0[CH_OPPONENT_HEADS, h1r, h1c] == 1.0
        assert obs0[CH_OPPONENT_HEADS, h2r, h2c] == 1.0

    def test_auxiliary_channels_include_opponents(self):
        engine = MultiplayerEngine(MatchConfig(player_count=2, seed=0))
        obs = encode_multi(engine.grid, engine.snakes, 0)

        assert 0.0 < obs[CH_OWN_LENGTH, 0, 0] < 1.0
        assert 0.0 < obs[CH_OPPONENT_MEAN_LENGTH, 0, 0] < 1.0
        assert obs[CH_DIR_RIGHT].sum() > 0.0
        assert 0.0 <= obs[CH_NEAREST_APPLE_DISTANCE, 0, 0] <= 1.0
        assert 0.0 <= obs[CH_NEAREST_DANGER_DISTANCE, 0, 0] <= 1.0

    def test_relative_mode_shifts_to_perspective_head(self):
        engine = MultiplayerEngine(MatchConfig(player_count=2, seed=0))
        abs_obs = encode_multi(engine.grid, engine.snakes, 0, mode="absolute")
        rel_obs = encode_multi(engine.grid, engine.snakes, 0, mode="relative")
        h, w = engine.config.effective_height, engine.config.effective_width

        own_head = engine.snakes[0].head
        opp_head = engine.snakes[1].head
        shift_r = (h // 2) - own_head[0]
        shift_c = (w // 2) - own_head[1]
        expected_opp = ((opp_head[0] + shift_r) % h, (opp_head[1] + shift_c) % w)

        assert abs_obs[CH_OWN_HEAD, own_head[0], own_head[1]] == 1.0
        assert rel_obs[CH_OWN_HEAD, h // 2, w // 2] == 1.0
        assert rel_obs[CH_OPPONENT_HEADS, expected_opp[0], expected_opp[1]] == 1.0
