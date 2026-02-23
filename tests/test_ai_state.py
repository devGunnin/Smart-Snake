"""Tests for state encoding."""

import numpy as np

from smart_snake.ai.state import (
    CH_APPLES,
    CH_OPPONENT_BODIES,
    CH_OPPONENT_HEADS,
    CH_OWN_BODY,
    CH_OWN_HEAD,
    CH_WALLS,
    NUM_CHANNELS,
    encode_multi,
    encode_single,
)
from smart_snake.engine import GameEngine
from smart_snake.grid import CellType, Grid
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
        assert obs[CH_WALLS].sum() == 2.0

    def test_dead_snake_empty_channels(self):
        grid = Grid(width=10, height=10)
        snake = Snake(5, 5, Direction.RIGHT, length=3)
        snake.alive = False
        obs = encode_single(grid, snake)
        assert obs[CH_OWN_HEAD].sum() == 0.0
        assert obs[CH_OWN_BODY].sum() == 0.0


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
