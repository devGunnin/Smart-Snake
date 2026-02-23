"""Gym-compatible environment wrappers for the snake engine.

Implements the Gymnasium ``reset()`` / ``step()`` API without depending
on the ``gymnasium`` package itself.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from smart_snake.ai.config import RewardConfig
from smart_snake.ai.state import NUM_CHANNELS, encode_multi, encode_single
from smart_snake.engine import GameEngine
from smart_snake.grid import WallMode
from smart_snake.multiplayer import MatchConfig, MultiplayerEngine
from smart_snake.snake import Direction

logger = logging.getLogger(__name__)

# Absolute action mapping: index → Direction.
ACTION_TO_DIRECTION: list[Direction] = [
    Direction.UP,
    Direction.DOWN,
    Direction.LEFT,
    Direction.RIGHT,
]
NUM_ACTIONS = len(ACTION_TO_DIRECTION)


@dataclass
class SpaceInfo:
    """Lightweight description of an observation or action space."""

    shape: tuple[int, ...]
    dtype: str
    n: int | None = None  # for discrete spaces


class SnakeEnv:
    """Single-player snake environment with Gymnasium-style API.

    Observations are ``(6, height, width)`` float32 arrays.
    Actions are integers in ``[0, 3]`` mapping to UP/DOWN/LEFT/RIGHT.
    """

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        wall_mode: str = "death",
        max_apples: int = 1,
        seed: int | None = None,
        reward_config: RewardConfig | None = None,
        max_steps: int = 1000,
    ) -> None:
        self._width = width
        self._height = height
        self._wall_mode_str = wall_mode
        self._wall_mode = WallMode(wall_mode)
        self._max_apples = max_apples
        self._seed = seed
        self._reward_cfg = reward_config or RewardConfig()
        self._max_steps = max_steps

        self.observation_space = SpaceInfo(
            shape=(NUM_CHANNELS, height, width), dtype="float32",
        )
        self.action_space = SpaceInfo(shape=(), dtype="int64", n=NUM_ACTIONS)

        self._engine: GameEngine | None = None
        self._steps = 0

    def reset(
        self, *, seed: int | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment and return ``(observation, info)``."""
        s = seed if seed is not None else self._seed
        self._engine = GameEngine(
            width=self._width,
            height=self._height,
            wall_mode=self._wall_mode,
            max_apples=self._max_apples,
            seed=s,
        )
        self._steps = 0
        obs = encode_single(self._engine.grid, self._engine.snake)
        return obs, {"score": 0, "tick": 0}

    def step(
        self, action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute *action* and return ``(obs, reward, terminated, truncated, info)``."""
        if self._engine is None:
            raise RuntimeError("Call reset() before step().")

        prev_score = self._engine.score
        self._engine.set_direction(ACTION_TO_DIRECTION[action])
        self._engine.step()
        self._steps += 1

        obs = encode_single(self._engine.grid, self._engine.snake)

        reward = self._reward_cfg.step_penalty
        if self._engine.score > prev_score:
            reward += self._reward_cfg.apple
        if self._engine.game_over:
            reward += self._reward_cfg.death

        terminated = self._engine.game_over
        truncated = (not terminated) and (self._steps >= self._max_steps)

        info = {
            "score": self._engine.score,
            "tick": self._engine.tick,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> str:
        """Return a text rendering of the current grid."""
        if self._engine is None:
            return ""
        symbols = {0: ".", 1: "S", 2: "A", 3: "#"}
        rows: list[str] = []
        for r in range(self._engine.grid.height):
            row_chars: list[str] = []
            for c in range(self._engine.grid.width):
                cell = int(self._engine.grid.cells[r, c])
                row_chars.append(symbols.get(cell, "?"))
            rows.append("".join(row_chars))
        return "\n".join(rows)


class MultiSnakeEnv:
    """Multi-agent snake environment for self-play training.

    Each call to :meth:`step` takes a list of actions (one per snake)
    and returns per-snake observations, rewards, and done flags.
    """

    def __init__(
        self,
        player_count: int = 2,
        grid_width: int | None = None,
        grid_height: int | None = None,
        wall_mode: str = "death",
        max_apples: int = 3,
        initial_snake_length: int = 3,
        seed: int | None = None,
        reward_config: RewardConfig | None = None,
        max_steps: int = 1000,
    ) -> None:
        self._player_count = player_count
        self._grid_width = grid_width
        self._grid_height = grid_height
        self._wall_mode_str = wall_mode
        self._max_apples = max_apples
        self._initial_snake_length = initial_snake_length
        self._seed = seed
        self._reward_cfg = reward_config or RewardConfig()
        self._max_steps = max_steps

        cfg = MatchConfig(
            player_count=player_count,
            grid_width=grid_width,
            grid_height=grid_height,
            wall_mode=WallMode(wall_mode),
            max_apples=max_apples,
            initial_snake_length=initial_snake_length,
        )
        h = cfg.effective_height
        w = cfg.effective_width

        self.num_agents = player_count
        self.observation_space = SpaceInfo(
            shape=(NUM_CHANNELS, h, w), dtype="float32",
        )
        self.action_space = SpaceInfo(shape=(), dtype="int64", n=NUM_ACTIONS)

        self._engine: MultiplayerEngine | None = None
        self._steps = 0
        self._prev_alive: list[bool] = []

    def reset(
        self, *, seed: int | None = None,
    ) -> tuple[list[np.ndarray], dict]:
        """Reset and return ``(observations, info)`` for all agents."""
        s = seed if seed is not None else self._seed
        self._engine = MultiplayerEngine(
            MatchConfig(
                player_count=self._player_count,
                grid_width=self._grid_width,
                grid_height=self._grid_height,
                wall_mode=WallMode(self._wall_mode_str),
                max_apples=self._max_apples,
                initial_snake_length=self._initial_snake_length,
                seed=s,
            ),
        )
        self._steps = 0
        self._prev_alive = [True] * self._player_count

        obs = [
            encode_multi(self._engine.grid, self._engine.snakes, i)
            for i in range(self._player_count)
        ]
        info = {"tick": 0, "scores": [0] * self._player_count}
        return obs, info

    def step(
        self, actions: list[int],
    ) -> tuple[list[np.ndarray], list[float], list[bool], list[bool], dict]:
        """Execute actions for all agents.

        Returns ``(observations, rewards, terminated, truncated, info)``.
        Each return value is a list with one entry per agent.
        """
        if self._engine is None:
            raise RuntimeError("Call reset() before step().")

        prev_scores = [p.score for p in self._engine.players]

        for sid in range(self._player_count):
            if self._engine.snakes[sid].alive:
                self._engine.set_direction(sid, ACTION_TO_DIRECTION[actions[sid]])
        self._engine.step()
        self._steps += 1

        obs: list[np.ndarray] = []
        rewards: list[float] = []
        terminated: list[bool] = []
        truncated_flags: list[bool] = []

        for sid in range(self._player_count):
            obs.append(
                encode_multi(self._engine.grid, self._engine.snakes, sid),
            )

            r = self._reward_cfg.step_penalty
            if self._engine.players[sid].score > prev_scores[sid]:
                r += self._reward_cfg.apple
            if self._prev_alive[sid] and not self._engine.snakes[sid].alive:
                r += self._reward_cfg.death
            if self._reward_cfg.survival_bonus and self._engine.snakes[sid].alive:
                r += self._reward_cfg.survival_bonus

            # Kill-opponent reward: count opponents that just died.
            if self._reward_cfg.kill_opponent:
                for oid in range(self._player_count):
                    if oid == sid:
                        continue
                    if (
                        self._prev_alive[oid]
                        and not self._engine.snakes[oid].alive
                        and self._engine.snakes[sid].alive
                    ):
                        r += self._reward_cfg.kill_opponent
            rewards.append(r)

            done = not self._engine.snakes[sid].alive or self._engine.game_over
            terminated.append(done)
            is_truncated = (
                not done and self._steps >= self._max_steps
            )
            truncated_flags.append(is_truncated)

        self._prev_alive = [s.alive for s in self._engine.snakes]

        info = {
            "tick": self._engine.tick,
            "scores": [p.score for p in self._engine.players],
            "game_over": self._engine.game_over,
            "winner": self._engine.winner,
        }
        return obs, rewards, terminated, truncated_flags, info
