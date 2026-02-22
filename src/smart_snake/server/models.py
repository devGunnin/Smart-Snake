"""Pydantic models for API request/response schemas."""

from __future__ import annotations

import enum

from pydantic import BaseModel, Field


class GameStatus(str, enum.Enum):
    """Lifecycle states for a game instance."""

    WAITING = "waiting"
    ACTIVE = "active"
    FINISHED = "finished"


class CreateGameRequest(BaseModel):
    """Request body for POST /games."""

    player_count: int = Field(default=2, ge=2, le=4)
    grid_width: int | None = Field(default=None, ge=4)
    grid_height: int | None = Field(default=None, ge=4)
    wall_mode: str = "death"
    max_apples: int = Field(default=3, ge=1)
    initial_snake_length: int = Field(default=3, ge=1)
    dead_body_mode: str = "remove"
    tick_rate_ms: int = Field(default=200, ge=50, le=2000)


class JoinRequest(BaseModel):
    """Request body for POST /games/{game_id}/join."""

    nickname: str = Field(default="player", min_length=1, max_length=32)


class StartRequest(BaseModel):
    """Request body for POST /games/{game_id}/start."""

    token: str


class GameSummary(BaseModel):
    """Compact game info for list endpoints."""

    game_id: str
    status: GameStatus
    player_count: int
    max_players: int
    tick_rate_ms: int


class JoinResponse(BaseModel):
    """Response for a successful lobby join."""

    game_id: str
    snake_id: int
    token: str
    nickname: str


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str
