"""In-memory game registry, lifecycle management, and async tick loops."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field

from starlette.websockets import WebSocket, WebSocketState

from smart_snake.grid import WallMode
from smart_snake.multiplayer import DeadBodyMode, MatchConfig, MultiplayerEngine
from smart_snake.server.models import GameStatus, GameSummary

logger = logging.getLogger(__name__)

# Simple rate limit: max games created per IP within the window.
_RATE_LIMIT_WINDOW = 60.0  # seconds
_RATE_LIMIT_MAX = 10
_RATE_COMPACT_INTERVAL = 60.0  # seconds between stale-key sweeps
_MAX_FINISHED_GAMES = 100


@dataclass
class PlayerSlot:
    """A reserved player slot in a game lobby."""

    snake_id: int
    nickname: str
    token: str
    websocket: WebSocket | None = None
    connected: bool = False


@dataclass
class GameInstance:
    """All state for a single game."""

    game_id: str
    config: MatchConfig
    tick_rate_ms: int
    status: GameStatus = GameStatus.WAITING
    engine: MultiplayerEngine | None = None
    players: dict[str, PlayerSlot] = field(default_factory=dict)
    spectators: list[WebSocket] = field(default_factory=list)
    host_token: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _task: asyncio.Task | None = field(default=None, repr=False)

    @property
    def player_count(self) -> int:
        return len(self.players)

    @property
    def max_players(self) -> int:
        return self.config.player_count


class GameManager:
    """Central registry managing all game instances."""

    def __init__(self, max_finished_games: int = _MAX_FINISHED_GAMES) -> None:
        if max_finished_games < 0:
            raise ValueError("max_finished_games must be >= 0.")
        self._games: dict[str, GameInstance] = {}
        self._rate_limits: dict[str, list[float]] = {}
        self._last_rate_compact: float = 0.0
        self._max_finished_games = max_finished_games

    def _check_rate_limit(self, client_ip: str) -> bool:
        """Return True if the client is within rate limits."""
        now = time.monotonic()
        timestamps = self._rate_limits.get(client_ip, [])
        timestamps = [t for t in timestamps if now - t < _RATE_LIMIT_WINDOW]
        if timestamps:
            self._rate_limits[client_ip] = timestamps
        else:
            self._rate_limits.pop(client_ip, None)
        self._compact_rate_limits(now)
        return len(timestamps) < _RATE_LIMIT_MAX

    def _compact_rate_limits(self, now: float) -> None:
        """Remove rate-limit entries whose timestamps have all expired."""
        if now - self._last_rate_compact < _RATE_COMPACT_INTERVAL:
            return
        self._last_rate_compact = now
        stale_ips = [
            ip for ip, ts in self._rate_limits.items()
            if all(now - t >= _RATE_LIMIT_WINDOW for t in ts)
        ]
        for ip in stale_ips:
            del self._rate_limits[ip]
        if stale_ips:
            logger.info(
                "Compacted %d stale rate-limit entries.", len(stale_ips),
            )

    def _record_creation(self, client_ip: str) -> None:
        self._rate_limits.setdefault(client_ip, []).append(time.monotonic())

    def create_game(
        self,
        player_count: int = 2,
        grid_width: int | None = None,
        grid_height: int | None = None,
        wall_mode: str = "death",
        max_apples: int = 3,
        initial_snake_length: int = 3,
        dead_body_mode: str = "remove",
        tick_rate_ms: int = 200,
        client_ip: str = "unknown",
    ) -> GameInstance:
        """Create a new game lobby and return the instance."""
        if not self._check_rate_limit(client_ip):
            raise ValueError("Rate limit exceeded. Try again later.")

        wm = WallMode(wall_mode)
        dbm = DeadBodyMode(dead_body_mode)
        config = MatchConfig(
            player_count=player_count,
            grid_width=grid_width,
            grid_height=grid_height,
            wall_mode=wm,
            max_apples=max_apples,
            initial_snake_length=initial_snake_length,
            dead_body_mode=dbm,
        )

        game_id = uuid.uuid4().hex[:12]
        instance = GameInstance(
            game_id=game_id,
            config=config,
            tick_rate_ms=tick_rate_ms,
        )
        self._games[game_id] = instance
        self._record_creation(client_ip)
        logger.info("Game %s created (players=%d).", game_id, player_count)
        return instance

    def get_game(self, game_id: str) -> GameInstance | None:
        return self._games.get(game_id)

    def list_games(self) -> list[GameSummary]:
        """Return summaries of non-finished games."""
        results: list[GameSummary] = []
        for g in self._games.values():
            if g.status == GameStatus.FINISHED:
                continue
            results.append(
                GameSummary(
                    game_id=g.game_id,
                    status=g.status,
                    player_count=g.player_count,
                    max_players=g.max_players,
                    tick_rate_ms=g.tick_rate_ms,
                )
            )
        return results

    def join_game(self, game_id: str, nickname: str) -> PlayerSlot:
        """Add a player to a waiting game lobby."""
        game = self._games.get(game_id)
        if game is None:
            raise KeyError(f"Game {game_id} not found.")
        if game.status != GameStatus.WAITING:
            raise ValueError("Game is not accepting players.")
        if game.player_count >= game.max_players:
            raise ValueError("Game lobby is full.")

        snake_id = game.player_count
        token = uuid.uuid4().hex
        slot = PlayerSlot(snake_id=snake_id, nickname=nickname, token=token)
        game.players[token] = slot

        if game.host_token is None:
            game.host_token = token

        logger.info(
            "Player '%s' joined game %s as snake %d.",
            nickname, game_id, snake_id,
        )
        return slot

    def start_game(self, game_id: str, token: str) -> None:
        """Start the game tick loop. Only the host can start."""
        game = self._games.get(game_id)
        if game is None:
            raise KeyError(f"Game {game_id} not found.")
        if game.status != GameStatus.WAITING:
            raise ValueError("Game is not in waiting state.")
        if game.player_count < 2:
            raise ValueError("Need at least 2 players to start.")
        if token != game.host_token:
            raise PermissionError("Only the host can start the game.")

        game.engine = MultiplayerEngine(game.config)
        game.status = GameStatus.ACTIVE
        game._task = asyncio.create_task(self._tick_loop(game))
        logger.info(
            "Game %s started with %d players.", game_id, game.player_count,
        )

    async def _tick_loop(self, game: GameInstance) -> None:
        """Run the game tick loop, broadcasting state each tick."""
        tick_interval = game.tick_rate_ms / 1000.0
        try:
            while game.status == GameStatus.ACTIVE:
                await asyncio.sleep(tick_interval)
                async with game.lock:
                    assert game.engine is not None  # noqa: S101
                    state = game.engine.step()
                    if game.engine.game_over:
                        self._mark_game_finished(game)
                await self._broadcast(game, state)
        except asyncio.CancelledError:
            logger.info("Tick loop cancelled for game %s.", game.game_id)
        except Exception:
            logger.exception("Tick loop error in game %s.", game.game_id)
            self._mark_game_finished(game)
        finally:
            if game.status == GameStatus.FINISHED:
                await self._close_connections(game)
                self._prune_finished_games()

    def _mark_game_finished(self, game: GameInstance) -> None:
        """Transition a game to finished exactly once."""
        if game.status != GameStatus.FINISHED:
            game.status = GameStatus.FINISHED
            game.finished_at = time.monotonic()

    async def _close_connections(self, game: GameInstance) -> None:
        """Close any live player and spectator sockets for a finished game."""
        for slot in game.players.values():
            ws = slot.websocket
            slot.websocket = None
            slot.connected = False
            if ws is None:
                continue
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.close(code=1000, reason="Game finished.")
            except Exception:
                logger.warning(
                    "Failed closing player socket for '%s' in game %s.",
                    slot.nickname,
                    game.game_id,
                )

        for ws in list(game.spectators):
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.close(code=1000, reason="Game finished.")
            except Exception:
                logger.warning("Failed closing spectator socket in game %s.", game.game_id)
        game.spectators.clear()

    def _prune_finished_games(self) -> None:
        """Bound retained finished games to avoid unbounded registry growth."""
        finished_games = [
            g for g in self._games.values() if g.status == GameStatus.FINISHED
        ]
        overflow = len(finished_games) - self._max_finished_games
        if overflow <= 0:
            return

        finished_games.sort(
            key=lambda g: g.finished_at if g.finished_at is not None else g.created_at,
        )
        for stale in finished_games[:overflow]:
            self._games.pop(stale.game_id, None)
        logger.info(
            "Pruned %d finished games (retaining up to %d).",
            overflow,
            self._max_finished_games,
        )

    async def _broadcast(self, game: GameInstance, state: dict) -> None:
        """Send game state to all connected players and spectators."""
        payload = json.dumps(state, separators=(",", ":"))
        dead_spectators: list[WebSocket] = []

        for slot in game.players.values():
            ws = slot.websocket
            if ws is None:
                continue
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(payload)
            except Exception:
                slot.websocket = None
                slot.connected = False

        # Iterate over a snapshot so concurrent disconnect handlers can mutate
        # the live spectator list without affecting this send loop.
        for ws in list(game.spectators):
            try:
                if ws.client_state == WebSocketState.CONNECTED:
                    await ws.send_text(payload)
            except Exception:
                dead_spectators.append(ws)

        for ws in dead_spectators:
            if ws in game.spectators:
                game.spectators.remove(ws)

    async def cleanup(self) -> None:
        """Cancel all running tick loops and release rate-limit state."""
        for game in self._games.values():
            if game._task and not game._task.done():
                game._task.cancel()
        tasks = [
            g._task for g in self._games.values()
            if g._task and not g._task.done()
        ]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._rate_limits.clear()
        logger.info("GameManager cleanup complete.")
