"""WebSocket handlers for real-time game play and spectating."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from smart_snake.server.game_manager import GameManager
from smart_snake.server.models import GameStatus
from smart_snake.snake import Direction

logger = logging.getLogger(__name__)

ws_router = APIRouter()

_DIRECTION_MAP: dict[str, Direction] = {
    "up": Direction.UP,
    "down": Direction.DOWN,
    "left": Direction.LEFT,
    "right": Direction.RIGHT,
}


def _get_manager(ws: WebSocket) -> GameManager:
    return ws.app.state.game_manager


@ws_router.websocket("/games/{game_id}/play")
async def play(websocket: WebSocket, game_id: str, token: str = "") -> None:
    """Player WebSocket: send directions, receive game state each tick."""
    manager = _get_manager(websocket)
    game = manager.get_game(game_id)
    if game is None:
        await websocket.close(code=4004, reason="Game not found.")
        return

    slot = game.players.get(token)
    if slot is None:
        await websocket.close(code=4001, reason="Invalid token.")
        return

    await websocket.accept()

    # Enforce a single active socket per player token.
    previous_ws = slot.websocket
    if previous_ws is not None and previous_ws is not websocket:
        try:
            await previous_ws.close(code=4008, reason="Replaced by new connection.")
        except Exception:
            logger.warning(
                "Failed closing previous socket for player '%s' in game %s.",
                slot.nickname,
                game_id,
            )

    slot.websocket = websocket
    slot.connected = True
    logger.info(
        "Player '%s' (snake %d) connected to game %s.",
        slot.nickname, slot.snake_id, game_id,
    )

    # Send initial state snapshot so the client gets immediate feedback.
    if game.engine is not None:
        state = game.engine.get_state()
        await websocket.send_text(
            json.dumps(state, separators=(",", ":")),
        )

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue

            direction_str = msg.get("direction")
            if not isinstance(direction_str, str):
                continue

            direction = _DIRECTION_MAP.get(direction_str.lower())
            if direction is None:
                continue

            async with game.lock:
                if (
                    game.engine is not None
                    and game.status == GameStatus.ACTIVE
                ):
                    game.engine.set_direction(slot.snake_id, direction)
    except WebSocketDisconnect:
        logger.info(
            "Player '%s' disconnected from game %s.",
            slot.nickname, game_id,
        )
    finally:
        # A newer connection may have replaced this socket while this handler
        # was still shutting down.
        if slot.websocket is websocket:
            slot.websocket = None
            slot.connected = False


@ws_router.websocket("/games/{game_id}/spectate")
async def spectate(websocket: WebSocket, game_id: str) -> None:
    """Spectator WebSocket: receive-only game state stream."""
    manager = _get_manager(websocket)
    game = manager.get_game(game_id)
    if game is None:
        await websocket.close(code=4004, reason="Game not found.")
        return

    await websocket.accept()
    game.spectators.append(websocket)
    logger.info("Spectator connected to game %s.", game_id)

    if game.engine is not None:
        state = game.engine.get_state()
        await websocket.send_text(
            json.dumps(state, separators=(",", ":")),
        )

    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        logger.info("Spectator disconnected from game %s.", game_id)
    finally:
        if websocket in game.spectators:
            game.spectators.remove(websocket)
