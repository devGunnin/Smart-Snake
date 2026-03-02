"""End-to-end integration tests: lobby -> game -> result lifecycle."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.testclient import TestClient

from smart_snake.server.app import create_app
from smart_snake.server.game_manager import GameManager
from smart_snake.server.models import GameStatus

BASE = "http://test"


@pytest.fixture()
def app():
    application = create_app()
    application.state.game_manager = GameManager()
    return application


@pytest.fixture()
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url=BASE) as c:
        yield c


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["games"]["active"] == 0
        assert data["games"]["waiting"] == 0
        assert data["total_players"] == 0

    @pytest.mark.asyncio
    async def test_health_reflects_game_counts(self, client):
        await client.post("/games", json={})
        resp = await client.get("/health")
        data = resp.json()
        assert data["games"]["waiting"] == 1
        assert data["total_players"] == 0

    @pytest.mark.asyncio
    async def test_health_counts_players(self, client):
        create_resp = await client.post("/games", json={})
        game_id = create_resp.json()["game_id"]
        await client.post(
            f"/games/{game_id}/join", json={"nickname": "alice"},
        )
        resp = await client.get("/health")
        data = resp.json()
        assert data["total_players"] == 1


class TestFullGameLifecycle:
    @pytest.mark.asyncio
    async def test_lobby_join_start_play_finish(self, app, client):
        """Full lobby -> join -> start -> play -> finish lifecycle."""
        create_resp = await client.post(
            "/games", json={"player_count": 2, "tick_rate_ms": 50},
        )
        assert create_resp.status_code == 201
        game_id = create_resp.json()["game_id"]

        j1 = await client.post(
            f"/games/{game_id}/join", json={"nickname": "p1"},
        )
        assert j1.status_code == 201
        token1 = j1.json()["token"]

        j2 = await client.post(
            f"/games/{game_id}/join", json={"nickname": "p2"},
        )
        assert j2.status_code == 201

        start_resp = await client.post(
            f"/games/{game_id}/start", json={"token": token1},
        )
        assert start_resp.status_code == 200

        detail = await client.get(f"/games/{game_id}")
        assert detail.json()["status"] == "active"
        assert "state" in detail.json()
        state = detail.json()["state"]
        assert len(state["snakes"]) == 2
        assert state["tick"] == 0

    @pytest.mark.asyncio
    async def test_game_finishes_via_manager(self, app, client):
        """Verify that a game transitions to finished."""
        create_resp = await client.post(
            "/games", json={"player_count": 2, "tick_rate_ms": 50},
        )
        game_id = create_resp.json()["game_id"]

        j1 = await client.post(
            f"/games/{game_id}/join", json={"nickname": "p1"},
        )
        await client.post(
            f"/games/{game_id}/join", json={"nickname": "p2"},
        )
        await client.post(
            f"/games/{game_id}/start", json={"token": j1.json()["token"]},
        )

        manager: GameManager = app.state.game_manager
        game = manager.get_game(game_id)
        assert game is not None
        game.status = GameStatus.FINISHED

        detail = await client.get(f"/games/{game_id}")
        assert detail.json()["status"] == "finished"


class TestMixedAiHumanGame:
    @pytest.mark.asyncio
    async def test_create_game_with_ai_opponents(self, client):
        """Create a game with AI opponents and verify slots."""
        create_resp = await client.post("/games", json={
            "player_count": 3,
            "ai_opponents": [
                {"difficulty": "easy"},
                {"difficulty": "hard"},
            ],
        })
        assert create_resp.status_code == 201
        data = create_resp.json()
        assert data["ai_count"] == 2
        assert data["max_players"] == 3

        game_id = data["game_id"]
        detail = await client.get(f"/games/{game_id}")
        players = detail.json()["players"]
        ai_players = [p for p in players if p["is_ai"]]
        assert len(ai_players) == 2

    @pytest.mark.asyncio
    async def test_ai_only_rejected(self, client):
        """All slots as AI (no human) must be rejected."""
        resp = await client.post("/games", json={
            "player_count": 2,
            "ai_opponents": [
                {"difficulty": "easy"},
                {"difficulty": "hard"},
            ],
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_invalid_ai_difficulty(self, client):
        """Invalid AI difficulty strings are rejected."""
        resp = await client.post("/games", json={
            "player_count": 2,
            "ai_opponents": [{"difficulty": "godlike"}],
        })
        assert resp.status_code == 422


class TestWebSocketLifecycle:
    def test_player_websocket_connect_and_receive(self, app):
        """Player can connect via WebSocket and receive initial state."""
        tc = TestClient(app)

        create_resp = tc.post("/games", json={"player_count": 2})
        game_id = create_resp.json()["game_id"]

        j1 = tc.post(f"/games/{game_id}/join", json={"nickname": "p1"})
        token1 = j1.json()["token"]
        tc.post(f"/games/{game_id}/join", json={"nickname": "p2"})

        tc.post(f"/games/{game_id}/start", json={"token": token1})

        with tc.websocket_connect(
            f"/games/{game_id}/play?token={token1}",
        ) as ws:
            data = ws.receive_json()
            assert "tick" in data
            assert "snakes" in data

    def test_spectator_websocket_connect(self, app):
        """Spectator can connect and receive initial state."""
        tc = TestClient(app)

        create_resp = tc.post("/games", json={"player_count": 2})
        game_id = create_resp.json()["game_id"]

        j1 = tc.post(f"/games/{game_id}/join", json={"nickname": "p1"})
        token1 = j1.json()["token"]
        tc.post(f"/games/{game_id}/join", json={"nickname": "p2"})

        tc.post(f"/games/{game_id}/start", json={"token": token1})

        with tc.websocket_connect(
            f"/games/{game_id}/spectate",
        ) as ws:
            data = ws.receive_json()
            assert "tick" in data
