"""Snake representation and movement logic."""

from __future__ import annotations

import enum
from collections import deque


class Direction(enum.Enum):
    """Cardinal movement directions with (row_delta, col_delta) values."""

    UP = (-1, 0)
    DOWN = (1, 0)
    LEFT = (0, -1)
    RIGHT = (0, 1)


# Pairs that would cause an instant 180° reversal.
_OPPOSITES: dict[Direction, Direction] = {
    Direction.UP: Direction.DOWN,
    Direction.DOWN: Direction.UP,
    Direction.LEFT: Direction.RIGHT,
    Direction.RIGHT: Direction.LEFT,
}


class Snake:
    """A snake represented as an ordered deque of (row, col) body segments.

    The head is ``body[0]``; the tail is ``body[-1]``.
    """

    def __init__(
        self,
        start_row: int,
        start_col: int,
        direction: Direction = Direction.RIGHT,
        length: int = 3,
    ) -> None:
        if length < 1:
            raise ValueError("Snake length must be at least 1.")
        dr, dc = direction.value
        self.body: deque[tuple[int, int]] = deque()
        for i in range(length):
            self.body.append((start_row - dr * i, start_col - dc * i))
        self.direction = direction
        self.alive = True
        self._grow_pending = 0

    @property
    def head(self) -> tuple[int, int]:
        """Return the head coordinate."""
        return self.body[0]

    def set_direction(self, new_direction: Direction) -> None:
        """Change direction, ignoring 180° reversals."""
        if _OPPOSITES.get(new_direction) != self.direction:
            self.direction = new_direction

    def next_head(self) -> tuple[int, int]:
        """Compute the next head position without moving."""
        dr, dc = self.direction.value
        r, c = self.head
        return r + dr, c + dc

    def advance(self, grow: bool = False) -> tuple[int, int] | None:
        """Move the snake one step forward.

        Returns the vacated tail cell, or ``None`` if the snake grew.
        """
        new_head = self.next_head()
        self.body.appendleft(new_head)
        if grow or self._grow_pending > 0:
            if self._grow_pending > 0:
                self._grow_pending -= 1
            return None
        return self.body.pop()

    def schedule_growth(self, segments: int = 1) -> None:
        """Queue growth for the next *segments* ticks."""
        self._grow_pending += segments

    def occupies(self, row: int, col: int) -> bool:
        """Check whether the snake occupies a given cell."""
        return (row, col) in self.body

    def self_collision(self) -> bool:
        """Check whether the head overlaps any other body segment."""
        head = self.head
        return any(seg == head for seg in list(self.body)[1:])

    def to_dict(self) -> dict:
        """Serialize snake state to a dictionary."""
        return {
            "body": [list(seg) for seg in self.body],
            "direction": self.direction.value,
            "alive": self.alive,
        }
