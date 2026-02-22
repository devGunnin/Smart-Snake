"""Step-based game engine composing grid, snake, and apple logic."""

from __future__ import annotations

import logging

import numpy as np

from smart_snake.apple import AppleSpawner
from smart_snake.grid import CellType, Grid, WallMode
from smart_snake.snake import Direction, Snake

logger = logging.getLogger(__name__)


class GameEngine:
    """Single-snake, step-based game engine.

    The engine owns the grid, snake, and apple spawner. Each call to
    :meth:`step` advances the game by one tick and returns the updated
    state dictionary.
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        wall_mode: WallMode = WallMode.DEATH,
        max_apples: int = 1,
        seed: int | None = None,
    ) -> None:
        self.grid = Grid(width=width, height=height, wall_mode=wall_mode)
        self.rng = np.random.default_rng(seed)

        start_row = height // 2
        start_col = width // 2
        self.snake = Snake(start_row, start_col, Direction.RIGHT, length=3)

        # Paint initial snake onto the grid.
        for r, c in self.snake.body:
            self.grid.set(r, c, CellType.SNAKE)

        self.apple_spawner = AppleSpawner(
            self.grid, max_apples=max_apples, rng=self.rng,
        )
        self.apple_spawner.spawn(max_apples)

        self.score = 0
        self.tick = 0
        self.game_over = False
        self._pending_direction: Direction | None = None

    def set_direction(self, direction: Direction) -> None:
        """Buffer at most one valid direction change for the next step."""
        if self._pending_direction is not None:
            return

        current_dr, current_dc = self.snake.direction.value
        next_dr, next_dc = direction.value
        is_reverse = (current_dr + next_dr == 0) and (current_dc + next_dc == 0)
        if is_reverse:
            return

        self._pending_direction = direction

    def step(self) -> dict:
        """Advance the game by one tick.

        Returns the full game state as a serializable dict.
        """
        if self.game_over:
            return self.get_state()

        if self._pending_direction is not None:
            self.snake.direction = self._pending_direction
            self._pending_direction = None

        next_r, next_c = self.snake.next_head()

        # --- boundary check ---
        if not self.grid.in_bounds(next_r, next_c):
            if self.grid.wall_mode == WallMode.WRAP:
                next_r, next_c = self.grid.wrap(next_r, next_c)
            else:
                self._kill_snake()
                return self.get_state()

        # --- self-collision check (look-ahead) ---
        # After advance the new head will be at (next_r, next_c).
        # We check against the body *excluding the tail* that will be removed
        # (unless the snake is about to grow).
        will_grow = self.grid.get(next_r, next_c) == CellType.APPLE

        # Check if next_head collides with any body segment that will remain.
        body_set = set(self.snake.body)
        if not will_grow:
            body_set.discard(self.snake.body[-1])  # tail will move away
        if (next_r, next_c) in body_set:
            self._kill_snake()
            return self.get_state()

        # --- move ---
        # Override next_head to use the (potentially wrapped) coordinate.
        self.snake.body.appendleft((next_r, next_c))
        vacated = None if will_grow else self.snake.body.pop()

        # Update grid cells and handle apple consumption.
        if will_grow:
            # Remove the consumed apple first; this sets the cell to EMPTY.
            # Repaint the head immediately after so the grid stays consistent
            # with the snake body before spawning any replacement apples.
            self.apple_spawner.remove(next_r, next_c)
            self.grid.set(next_r, next_c, CellType.SNAKE)
            self.score += 1
            self.apple_spawner.spawn(1)
        else:
            self.grid.set(next_r, next_c, CellType.SNAKE)
            self.grid.set(vacated[0], vacated[1], CellType.EMPTY)

        self.tick += 1
        return self.get_state()

    def get_state(self) -> dict:
        """Return the full, serializable game state."""
        return {
            "tick": self.tick,
            "score": self.score,
            "game_over": self.game_over,
            "grid": self.grid.to_dict(),
            "snake": self.snake.to_dict(),
            "apples": self.apple_spawner.to_dict(),
        }

    def _kill_snake(self) -> None:
        """Mark the snake as dead and end the game."""
        self.snake.alive = False
        self.game_over = True
        self.tick += 1
        logger.info("Snake died at tick %d with score %d.", self.tick, self.score)
