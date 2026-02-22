"""Smart Snake — core game engine."""

from smart_snake.engine import GameEngine
from smart_snake.grid import Grid, WallMode
from smart_snake.multiplayer import (
    DeadBodyMode,
    MatchConfig,
    MultiplayerEngine,
    PlayerState,
)
from smart_snake.snake import Direction, Snake

__all__ = [
    "DeadBodyMode",
    "Direction",
    "GameEngine",
    "Grid",
    "MatchConfig",
    "MultiplayerEngine",
    "PlayerState",
    "Snake",
    "WallMode",
]
