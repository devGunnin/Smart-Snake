"""Multiplayer game engine supporting 2–4 simultaneous snakes."""

from __future__ import annotations

import enum
import logging
from collections import Counter
from dataclasses import dataclass

import numpy as np

from smart_snake.apple import AppleSpawner
from smart_snake.grid import CellType, Grid, WallMode
from smart_snake.snake import Direction, Snake

logger = logging.getLogger(__name__)

# Default grid dimensions per player count.
_DEFAULT_GRID_SIZES: dict[int, tuple[int, int]] = {
    2: (20, 20),
    3: (25, 25),
    4: (30, 30),
}

# Spawn configs: (row_fraction, col_fraction, direction) per player slot.
_SPAWN_LAYOUT: list[tuple[float, float, Direction]] = [
    (0.25, 0.25, Direction.RIGHT),
    (0.75, 0.75, Direction.LEFT),
    (0.25, 0.75, Direction.DOWN),
    (0.75, 0.25, Direction.UP),
]


class DeadBodyMode(enum.Enum):
    """Determines what happens to a snake's body after it dies."""

    REMOVE = "remove"
    OBSTACLE = "obstacle"


@dataclass(frozen=True)
class MatchConfig:
    """Configuration for a multiplayer match."""

    player_count: int = 2
    grid_width: int | None = None
    grid_height: int | None = None
    wall_mode: WallMode = WallMode.DEATH
    max_apples: int = 3
    initial_snake_length: int = 3
    dead_body_mode: DeadBodyMode = DeadBodyMode.REMOVE
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 2 <= self.player_count <= 4:
            raise ValueError("player_count must be between 2 and 4.")
        if self.max_apples < 1:
            raise ValueError("max_apples must be at least 1.")
        if self.initial_snake_length < 1:
            raise ValueError("initial_snake_length must be at least 1.")

        width = self.effective_width
        height = self.effective_height
        if width < 4 or height < 4:
            raise ValueError("grid_width and grid_height must each be at least 4.")

        occupied: set[tuple[int, int]] = set()
        for i in range(self.player_count):
            row_frac, col_frac, direction = _SPAWN_LAYOUT[i]
            row = int(height * row_frac)
            col = int(width * col_frac)
            dr, dc = direction.value

            for seg in range(self.initial_snake_length):
                r = row - dr * seg
                c = col - dc * seg
                if not (0 <= r < height and 0 <= c < width):
                    raise ValueError(
                        "initial_snake_length does not fit the configured grid "
                        f"for snake {i}; increase grid size or reduce length."
                    )
                if (r, c) in occupied:
                    raise ValueError(
                        "spawn layout overlaps for this configuration; increase "
                        "grid size or reduce initial_snake_length."
                    )
                occupied.add((r, c))

    @property
    def effective_width(self) -> int:
        if self.grid_width is not None:
            return self.grid_width
        return _DEFAULT_GRID_SIZES[self.player_count][0]

    @property
    def effective_height(self) -> int:
        if self.grid_height is not None:
            return self.grid_height
        return _DEFAULT_GRID_SIZES[self.player_count][1]


class PlayerState:
    """Tracks per-snake scoring and status."""

    __slots__ = ("snake_id", "score", "survival_ticks", "alive")

    def __init__(self, snake_id: int) -> None:
        self.snake_id = snake_id
        self.score = 0
        self.survival_ticks = 0
        self.alive = True

    def to_dict(self) -> dict:
        return {
            "snake_id": self.snake_id,
            "score": self.score,
            "survival_ticks": self.survival_ticks,
            "alive": self.alive,
        }


class MultiplayerEngine:
    """Step-based game engine for 2–4 simultaneous snakes.

    Each call to :meth:`step` advances the game by one tick, resolving
    all collisions simultaneously and returning the full game state.
    """

    def __init__(self, config: MatchConfig | None = None) -> None:
        cfg = config or MatchConfig()
        self.config = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.grid = Grid(
            width=cfg.effective_width,
            height=cfg.effective_height,
            wall_mode=cfg.wall_mode,
        )

        self.snakes: list[Snake] = []
        self.players: list[PlayerState] = []
        self._pending_directions: list[Direction | None] = []

        self._spawn_snakes(cfg)

        self.apple_spawner = AppleSpawner(
            self.grid, max_apples=cfg.max_apples, rng=self.rng,
        )
        self.apple_spawner.spawn(cfg.max_apples)

        self.tick = 0
        self.game_over = False
        self.winner: int | None = None

    def _spawn_snakes(self, cfg: MatchConfig) -> None:
        """Place snakes in spread-out positions across the grid."""
        h, w = cfg.effective_height, cfg.effective_width
        for i in range(cfg.player_count):
            row_frac, col_frac, direction = _SPAWN_LAYOUT[i]
            row = int(h * row_frac)
            col = int(w * col_frac)
            snake = Snake(row, col, direction, length=cfg.initial_snake_length)
            for r, c in snake.body:
                self.grid.set(r, c, CellType.SNAKE)
            self.snakes.append(snake)
            self.players.append(PlayerState(i))
            self._pending_directions.append(None)

    def set_direction(self, snake_id: int, direction: Direction) -> None:
        """Buffer a direction change for a specific snake."""
        if not 0 <= snake_id < len(self.snakes):
            raise ValueError(
                f"snake_id {snake_id} out of range [0, {len(self.snakes)})."
            )
        if not self.snakes[snake_id].alive:
            return
        if self._pending_directions[snake_id] is not None:
            return

        current = self.snakes[snake_id].direction
        dr_cur, dc_cur = current.value
        dr_new, dc_new = direction.value
        if (dr_cur + dr_new == 0) and (dc_cur + dc_new == 0):
            return
        self._pending_directions[snake_id] = direction

    def step(self) -> dict:
        """Advance the game by one tick with simultaneous collision resolution."""
        if self.game_over:
            return self.get_state()

        # Apply buffered directions.
        for i, pending in enumerate(self._pending_directions):
            if pending is not None:
                self.snakes[i].direction = pending
            self._pending_directions[i] = None

        alive_ids = [i for i, s in enumerate(self.snakes) if s.alive]

        # Compute next heads for all alive snakes.
        next_heads: dict[int, tuple[int, int]] = {}
        wall_dead: set[int] = set()

        for sid in alive_ids:
            nr, nc = self.snakes[sid].next_head()
            if not self.grid.in_bounds(nr, nc):
                if self.grid.wall_mode == WallMode.WRAP:
                    nr, nc = self.grid.wrap(nr, nc)
                else:
                    wall_dead.add(sid)
                    continue
            next_heads[sid] = (nr, nc)

        # Determine growth intent from pre-move grid state.
        will_grow_by_sid: dict[int, bool] = {}
        for sid, head_pos in next_heads.items():
            will_grow_by_sid[sid] = (
                self.grid.in_bounds(head_pos[0], head_pos[1])
                and self.grid.get(head_pos[0], head_pos[1]) == CellType.APPLE
            )

        # Detect head-to-head collisions.
        head_counts: Counter[tuple[int, int]] = Counter()
        for sid, pos in next_heads.items():
            if sid not in wall_dead:
                head_counts[pos] += 1

        head_head_dead: set[int] = set()
        for sid, pos in next_heads.items():
            if sid not in wall_dead and head_counts[pos] > 1:
                head_head_dead.add(sid)

        # Detect head-to-body, self, and obstacle collisions.
        # Snakes that die this tick do not move, so their tails remain occupied.
        # Resolve to a fixed point because each newly dead snake can expose
        # further collisions against a tail that will no longer vacate.
        obstacle_dead: set[int] = set()
        for sid, pos in next_heads.items():
            if sid in wall_dead or sid in head_head_dead:
                continue
            if self.grid.get(pos[0], pos[1]) == CellType.OBSTACLE:
                obstacle_dead.add(sid)

        immobile: set[int] = set(wall_dead | head_head_dead | obstacle_dead)
        while True:
            occupation: set[tuple[int, int]] = set()
            for sid in alive_ids:
                snake = self.snakes[sid]
                occupation.update(snake.body)
                if sid in immobile or sid not in next_heads:
                    continue
                if not will_grow_by_sid.get(sid, False) and snake.body:
                    occupation.discard(snake.body[-1])

            new_body_dead: set[int] = set()
            for sid, pos in next_heads.items():
                if sid in immobile:
                    continue
                if pos in occupation:
                    new_body_dead.add(sid)

            if not new_body_dead:
                break
            immobile.update(new_body_dead)

        body_dead = immobile - wall_dead - head_head_dead

        # Combine all deaths for this tick.
        all_dead = wall_dead | head_head_dead | body_dead

        # Kill dead snakes.
        for sid in all_dead:
            self._kill_snake(sid)

        # Move surviving snakes.
        survivors = [
            sid for sid in alive_ids
            if sid not in all_dead and sid in next_heads
        ]
        consumed_apples: list[tuple[int, int]] = []
        vacated_cells: list[tuple[int, int]] = []
        for sid in survivors:
            snake = self.snakes[sid]
            nr, nc = next_heads[sid]
            will_grow = will_grow_by_sid.get(sid, False)

            snake.body.appendleft((nr, nc))
            if will_grow:
                consumed_apples.append((nr, nc))
                self.players[sid].score += 1
            else:
                vacated_cells.append(snake.body.pop())

        for apple_r, apple_c in consumed_apples:
            self.apple_spawner.remove(apple_r, apple_c)

        occupied_after_move: set[tuple[int, int]] = set()
        for sid in survivors:
            occupied_after_move.update(self.snakes[sid].body)

        for tail_r, tail_c in vacated_cells:
            if (tail_r, tail_c) in occupied_after_move:
                continue
            if self.grid.get(tail_r, tail_c) == CellType.SNAKE:
                self.grid.set(tail_r, tail_c, CellType.EMPTY)

        for sid in survivors:
            head_r, head_c = self.snakes[sid].head
            self.grid.set(head_r, head_c, CellType.SNAKE)

        if consumed_apples:
            self.apple_spawner.spawn(len(consumed_apples))

        # Update survival ticks for living snakes.
        self.tick += 1
        for sid in alive_ids:
            if self.snakes[sid].alive:
                self.players[sid].survival_ticks = self.tick

        # Check win condition.
        remaining = [i for i, s in enumerate(self.snakes) if s.alive]
        if len(remaining) <= 1:
            self.game_over = True
            if len(remaining) == 1:
                self.winner = remaining[0]
            elif len(all_dead) >= 2:
                # Tie-break: longest snake at time of mutual death.
                best_id = max(
                    all_dead,
                    key=lambda sid: len(self.snakes[sid].body),
                )
                tied = [
                    sid for sid in all_dead
                    if len(self.snakes[sid].body) == len(
                        self.snakes[best_id].body
                    )
                ]
                self.winner = best_id if len(tied) == 1 else None

        return self.get_state()

    def _kill_snake(self, snake_id: int) -> None:
        """Mark a snake as dead and update the grid."""
        snake = self.snakes[snake_id]
        snake.alive = False
        self.players[snake_id].alive = False

        if self.config.dead_body_mode == DeadBodyMode.REMOVE:
            for r, c in snake.body:
                if self.grid.get(r, c) == CellType.SNAKE:
                    self.grid.set(r, c, CellType.EMPTY)
        else:
            for r, c in snake.body:
                if self.grid.get(r, c) == CellType.SNAKE:
                    self.grid.set(r, c, CellType.OBSTACLE)

        logger.info(
            "Snake %d died at tick %d with score %d.",
            snake_id,
            self.tick + 1,
            self.players[snake_id].score,
        )

    def get_state(self) -> dict:
        """Return the full, serializable game state."""
        return {
            "tick": self.tick,
            "game_over": self.game_over,
            "winner": self.winner,
            "grid": self.grid.to_dict(),
            "snakes": [s.to_dict() for s in self.snakes],
            "players": [p.to_dict() for p in self.players],
            "apples": self.apple_spawner.to_dict(),
            "config": {
                "player_count": self.config.player_count,
                "grid_width": self.config.effective_width,
                "grid_height": self.config.effective_height,
                "wall_mode": self.config.wall_mode.value,
                "max_apples": self.config.max_apples,
                "initial_snake_length": self.config.initial_snake_length,
                "dead_body_mode": self.config.dead_body_mode.value,
            },
        }
