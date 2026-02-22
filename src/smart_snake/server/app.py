"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from smart_snake.server.game_manager import GameManager
from smart_snake.server.routes import router
from smart_snake.server.websocket import ws_router


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.game_manager = GameManager()
    yield
    await app.state.game_manager.cleanup()


def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="Smart Snake API", version="0.1.0", lifespan=_lifespan,
    )
    app.include_router(router)
    app.include_router(ws_router)
    return app
