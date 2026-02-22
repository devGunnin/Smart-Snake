"""REST API route handlers for game lifecycle management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from smart_snake.server.models import (
    CreateGameRequest,
    GameSummary,
    JoinRequest,
    JoinResponse,
    StartRequest,
)

router = APIRouter(prefix="/games", tags=["games"])


def _get_manager(request: Request):
    return request.app.state.game_manager


@router.post("", status_code=201)
async def create_game(body: CreateGameRequest, request: Request) -> GameSummary:
    """Create a new game lobby."""
    manager = _get_manager(request)
    client_ip = request.client.host if request.client else "unknown"
    try:
        game = manager.create_game(
            player_count=body.player_count,
            grid_width=body.grid_width,
            grid_height=body.grid_height,
            wall_mode=body.wall_mode,
            max_apples=body.max_apples,
            initial_snake_length=body.initial_snake_length,
            dead_body_mode=body.dead_body_mode,
            tick_rate_ms=body.tick_rate_ms,
            client_ip=client_ip,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return GameSummary(
        game_id=game.game_id,
        status=game.status,
        player_count=game.player_count,
        max_players=game.max_players,
        tick_rate_ms=game.tick_rate_ms,
    )


@router.get("")
async def list_games(request: Request) -> list[GameSummary]:
    """List active and waiting games."""
    return _get_manager(request).list_games()


@router.get("/{game_id}")
async def get_game(game_id: str, request: Request) -> dict:
    """Get full game state/metadata."""
    manager = _get_manager(request)
    game = manager.get_game(game_id)
    if game is None:
        raise HTTPException(status_code=404, detail="Game not found.")
    result: dict = {
        "game_id": game.game_id,
        "status": game.status.value,
        "player_count": game.player_count,
        "max_players": game.max_players,
        "tick_rate_ms": game.tick_rate_ms,
        "players": [
            {
                "snake_id": s.snake_id,
                "nickname": s.nickname,
                "connected": s.connected,
            }
            for s in game.players.values()
        ],
    }
    if game.engine is not None:
        result["state"] = game.engine.get_state()
    return result


@router.post("/{game_id}/join", status_code=201)
async def join_game(
    game_id: str, body: JoinRequest, request: Request,
) -> JoinResponse:
    """Join a game lobby."""
    manager = _get_manager(request)
    try:
        slot = manager.join_game(game_id, body.nickname)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return JoinResponse(
        game_id=game_id,
        snake_id=slot.snake_id,
        token=slot.token,
        nickname=slot.nickname,
    )


@router.post("/{game_id}/start", status_code=200)
async def start_game(
    game_id: str, body: StartRequest, request: Request,
) -> dict:
    """Start the game (host only)."""
    manager = _get_manager(request)
    try:
        manager.start_game(game_id, body.token)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    return {"status": "started", "game_id": game_id}
