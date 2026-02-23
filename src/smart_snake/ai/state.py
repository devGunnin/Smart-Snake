"""State encoding: convert game state to multi-channel NumPy arrays."""

from __future__ import annotations

import numpy as np

from smart_snake.grid import CellType, Grid
from smart_snake.snake import Snake

# Channel indices for the 6-channel observation.
CH_OWN_HEAD = 0
CH_OWN_BODY = 1
CH_OPPONENT_HEADS = 2
CH_OPPONENT_BODIES = 3
CH_APPLES = 4
CH_WALLS = 5
NUM_CHANNELS = 6


def encode_single(
    grid: Grid,
    snake: Snake,
) -> np.ndarray:
    """Encode a single-player game state as a ``(6, H, W)`` float32 array.

    Channels 2–3 (opponent heads/bodies) are all-zero in single-player mode.
    """
    h, w = grid.height, grid.width
    obs = np.zeros((NUM_CHANNELS, h, w), dtype=np.float32)

    if snake.alive and snake.body:
        hr, hc = snake.head
        obs[CH_OWN_HEAD, hr, hc] = 1.0
        for r, c in snake.body:
            obs[CH_OWN_BODY, r, c] = 1.0

    cells = grid.cells
    obs[CH_APPLES] = (cells == CellType.APPLE).astype(np.float32)
    obs[CH_WALLS] = (cells == CellType.OBSTACLE).astype(np.float32)

    return obs


def encode_multi(
    grid: Grid,
    snakes: list[Snake],
    perspective_id: int,
) -> np.ndarray:
    """Encode a multiplayer game state from one snake's perspective.

    Returns a ``(6, H, W)`` float32 array where *own* channels refer to
    the snake identified by *perspective_id* and *opponent* channels
    aggregate all other alive snakes.
    """
    h, w = grid.height, grid.width
    obs = np.zeros((NUM_CHANNELS, h, w), dtype=np.float32)

    own = snakes[perspective_id]
    if own.alive and own.body:
        hr, hc = own.head
        obs[CH_OWN_HEAD, hr, hc] = 1.0
        for r, c in own.body:
            obs[CH_OWN_BODY, r, c] = 1.0

    for sid, snake in enumerate(snakes):
        if sid == perspective_id or not snake.alive or not snake.body:
            continue
        ohr, ohc = snake.head
        obs[CH_OPPONENT_HEADS, ohr, ohc] = 1.0
        for r, c in snake.body:
            obs[CH_OPPONENT_BODIES, r, c] = 1.0

    cells = grid.cells
    obs[CH_APPLES] = (cells == CellType.APPLE).astype(np.float32)
    obs[CH_WALLS] = (cells == CellType.OBSTACLE).astype(np.float32)

    return obs
