"""State encoding: convert game state to multi-channel NumPy arrays."""

from __future__ import annotations

from typing import Literal

import numpy as np

from smart_snake.grid import CellType, Grid, WallMode
from smart_snake.snake import Direction, Snake

# Channel indices for spatial observation channels.
CH_OWN_HEAD = 0
CH_OWN_BODY = 1
CH_OPPONENT_HEADS = 2
CH_OPPONENT_BODIES = 3
CH_APPLES = 4
CH_WALLS = 5
NUM_SPATIAL_CHANNELS = 6

# Auxiliary feature channels.
CH_OWN_LENGTH = NUM_SPATIAL_CHANNELS
CH_OPPONENT_MEAN_LENGTH = NUM_SPATIAL_CHANNELS + 1
CH_DIR_UP = NUM_SPATIAL_CHANNELS + 2
CH_DIR_DOWN = NUM_SPATIAL_CHANNELS + 3
CH_DIR_LEFT = NUM_SPATIAL_CHANNELS + 4
CH_DIR_RIGHT = NUM_SPATIAL_CHANNELS + 5
CH_NEAREST_APPLE_DISTANCE = NUM_SPATIAL_CHANNELS + 6
CH_NEAREST_DANGER_DISTANCE = NUM_SPATIAL_CHANNELS + 7
NUM_AUX_CHANNELS = 8
NUM_CHANNELS = NUM_SPATIAL_CHANNELS + NUM_AUX_CHANNELS

StateEncodingMode = Literal["absolute", "relative"]

_VALID_MODES: set[StateEncodingMode] = {"absolute", "relative"}
_DIRECTION_CHANNEL_BY_VALUE = {
    Direction.UP.value: CH_DIR_UP,
    Direction.DOWN.value: CH_DIR_DOWN,
    Direction.LEFT.value: CH_DIR_LEFT,
    Direction.RIGHT.value: CH_DIR_RIGHT,
}


def _validate_mode(mode: str) -> StateEncodingMode:
    if mode not in _VALID_MODES:
        raise ValueError(
            f"Unsupported state encoding mode: {mode!r}. "
            f"Expected one of {sorted(_VALID_MODES)}.",
        )
    return mode


def _encode_spatial_absolute(
    grid: Grid,
    snakes: list[Snake],
    perspective_id: int,
) -> np.ndarray:
    """Encode only spatial channels in absolute grid coordinates."""
    h, w = grid.height, grid.width
    obs = np.zeros((NUM_SPATIAL_CHANNELS, h, w), dtype=np.float32)

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

    wall_mask = cells == CellType.OBSTACLE
    if grid.wall_mode == WallMode.DEATH:
        wall_mask = wall_mask.copy()
        wall_mask[0, :] = True
        wall_mask[-1, :] = True
        wall_mask[:, 0] = True
        wall_mask[:, -1] = True
    obs[CH_WALLS] = wall_mask.astype(np.float32)
    return obs


def _head_centered_relative(
    spatial_obs: np.ndarray,
    head: tuple[int, int] | None,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Shift the spatial tensor so the perspective snake head is centered."""
    if head is None:
        return spatial_obs
    target_row = height // 2
    target_col = width // 2
    shift = (target_row - head[0], target_col - head[1])
    return np.roll(spatial_obs, shift=shift, axis=(1, 2))


def _normalized_min_distance(
    source: tuple[int, int] | None,
    targets: set[tuple[int, int]],
    *,
    height: int,
    width: int,
) -> float:
    """Return normalized Manhattan distance to nearest target in [0, 1]."""
    max_distance = max(height + width - 2, 1)
    if source is None or not targets:
        return 1.0
    sr, sc = source
    nearest = min(abs(sr - tr) + abs(sc - tc) for tr, tc in targets)
    return float(min(nearest / max_distance, 1.0))


def _danger_cells(
    grid: Grid,
    snakes: list[Snake],
    perspective_id: int,
) -> set[tuple[int, int]]:
    """Cells that are currently dangerous for the perspective snake."""
    danger: set[tuple[int, int]] = set()

    if grid.wall_mode == WallMode.DEATH:
        for r in range(grid.height):
            danger.add((r, 0))
            danger.add((r, grid.width - 1))
        for c in range(grid.width):
            danger.add((0, c))
            danger.add((grid.height - 1, c))

    rows, cols = np.where(grid.cells == CellType.OBSTACLE)
    danger.update(zip(rows.tolist(), cols.tolist(), strict=True))

    for sid, snake in enumerate(snakes):
        if not snake.alive or not snake.body:
            continue
        for idx, seg in enumerate(snake.body):
            if sid == perspective_id and idx == 0:
                continue
            if sid == perspective_id and idx == 1:
                # Immediate reversal is disallowed, so the neck is not reachable.
                continue
            danger.add(seg)
    return danger


def _apple_cells(grid: Grid) -> set[tuple[int, int]]:
    rows, cols = np.where(grid.cells == CellType.APPLE)
    return set(zip(rows.tolist(), cols.tolist(), strict=True))


def _append_auxiliary_channels(
    spatial_obs: np.ndarray,
    grid: Grid,
    snakes: list[Snake],
    perspective_id: int,
) -> np.ndarray:
    """Append non-spatial feature channels to spatial observations."""
    h, w = grid.height, grid.width
    obs = np.zeros((NUM_CHANNELS, h, w), dtype=np.float32)
    obs[:NUM_SPATIAL_CHANNELS] = spatial_obs

    own = snakes[perspective_id]
    own_length = len(own.body) if own.alive else 0
    opponent_lengths = [
        len(s.body)
        for sid, s in enumerate(snakes)
        if sid != perspective_id and s.alive
    ]
    length_scale = float(max(h * w, 1))
    own_length_norm = own_length / length_scale
    opponent_mean_norm = (
        (sum(opponent_lengths) / len(opponent_lengths)) / length_scale
        if opponent_lengths else 0.0
    )

    obs[CH_OWN_LENGTH] = own_length_norm
    obs[CH_OPPONENT_MEAN_LENGTH] = opponent_mean_norm

    if own.alive:
        dir_ch = _DIRECTION_CHANNEL_BY_VALUE[own.direction.value]
        obs[dir_ch] = 1.0

    head = own.head if own.alive and own.body else None
    apple_dist = _normalized_min_distance(
        head, _apple_cells(grid), height=h, width=w,
    )
    danger_dist = _normalized_min_distance(
        head, _danger_cells(grid, snakes, perspective_id), height=h, width=w,
    )
    obs[CH_NEAREST_APPLE_DISTANCE] = apple_dist
    obs[CH_NEAREST_DANGER_DISTANCE] = danger_dist
    return obs


def encode_single(
    grid: Grid,
    snake: Snake,
    mode: StateEncodingMode = "absolute",
) -> np.ndarray:
    """Encode a single-player game state as a ``(C, H, W)`` float32 array.

    Channels 2–3 (opponent heads/bodies) are all-zero in single-player mode.
    """
    enc_mode = _validate_mode(mode)
    snakes = [snake]
    spatial = _encode_spatial_absolute(grid, snakes, perspective_id=0)
    if enc_mode == "relative":
        head = snake.head if snake.alive and snake.body else None
        spatial = _head_centered_relative(
            spatial,
            head,
            height=grid.height,
            width=grid.width,
        )
    return _append_auxiliary_channels(spatial, grid, snakes, perspective_id=0)


def encode_multi(
    grid: Grid,
    snakes: list[Snake],
    perspective_id: int,
    mode: StateEncodingMode = "absolute",
) -> np.ndarray:
    """Encode a multiplayer game state from one snake's perspective.

    Returns a ``(C, H, W)`` float32 array where *own* channels refer to
    the snake identified by *perspective_id* and *opponent* channels
    aggregate all other alive snakes.
    """
    enc_mode = _validate_mode(mode)
    spatial = _encode_spatial_absolute(grid, snakes, perspective_id=perspective_id)
    if enc_mode == "relative":
        own = snakes[perspective_id]
        head = own.head if own.alive and own.body else None
        spatial = _head_centered_relative(
            spatial,
            head,
            height=grid.height,
            width=grid.width,
        )
    return _append_auxiliary_channels(
        spatial, grid, snakes, perspective_id=perspective_id,
    )
