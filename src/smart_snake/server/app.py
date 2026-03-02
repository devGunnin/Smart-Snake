"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from smart_snake.server.game_manager import GameManager
from smart_snake.server.routes import router
from smart_snake.server.websocket import ws_router

_FRONTEND_DIR = "frontend/dist"


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.game_manager = GameManager()
    yield
    await app.state.game_manager.cleanup()


def create_app(*, serve_frontend: bool = False) -> FastAPI:
    """Build and return the FastAPI application."""
    app = FastAPI(
        title="Smart Snake API", version="0.1.0", lifespan=_lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    app.include_router(ws_router)

    if serve_frontend:
        from pathlib import Path

        dist = Path(_FRONTEND_DIR)
        if dist.is_dir():
            app.mount(
                "/", StaticFiles(directory=str(dist), html=True),
            )

    return app
