"""REST API endpoint tests."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from smart_snake.server.app import create_app
from smart_snake.server.game_manager import GameManager

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


class TestCreateGame:
    @pytest.mark.asyncio
    async def test_create_default(self, client):
        resp = await client.post("/games", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "waiting"
        assert data["max_players"] == 2
        assert data["player_count"] == 0
        assert "game_id" in data

    @pytest.mark.asyncio
    async def test_create_4_player(self, client):
        resp = await client.post("/games", json={"player_count": 4})
        assert resp.status_code == 201
        assert resp.json()["max_players"] == 4

    @pytest.mark.asyncio
    async def test_create_invalid_player_count(self, client):
        resp = await client.post("/games", json={"player_count": 5})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_invalid_wall_mode(self, client):
        resp = await client.post("/games", json={"wall_mode": "invalid"})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_create_custom_config(self, client):
        resp = await client.post("/games", json={
            "player_count": 3,
            "grid_width": 25,
            "grid_height": 25,
            "wall_mode": "wrap",
            "max_apples": 5,
            "tick_rate_ms": 100,
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["max_players"] == 3
        assert data["tick_rate_ms"] == 100


class TestListGames:
    @pytest.mark.asyncio
    async def test_list_empty(self, client):
        resp = await client.get("/games")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_list_after_create(self, client):
        await client.post("/games", json={})
        resp = await client.get("/games")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["status"] == "waiting"


class TestGetGame:
    @pytest.mark.asyncio
    async def test_get_existing(self, client):
        create_resp = await client.post("/games", json={})
        game_id = create_resp.json()["game_id"]
        resp = await client.get(f"/games/{game_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["game_id"] == game_id
        assert data["status"] == "waiting"

    @pytest.mark.asyncio
    async def test_get_not_found(self, client):
        resp = await client.get("/games/nonexistent")
        assert resp.status_code == 404


class TestJoinGame:
    @pytest.mark.asyncio
    async def test_join_success(self, client):
        create_resp = await client.post("/games", json={})
        game_id = create_resp.json()["game_id"]

        resp = await client.post(
            f"/games/{game_id}/join", json={"nickname": "alice"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["snake_id"] == 0
        assert data["nickname"] == "alice"
        assert "token" in data

    @pytest.mark.asyncio
    async def test_join_fills_lobby(self, client):
        create_resp = await client.post(
            "/games", json={"player_count": 2},
        )
        game_id = create_resp.json()["game_id"]

        await client.post(
            f"/games/{game_id}/join", json={"nickname": "p1"},
        )
        await client.post(
            f"/games/{game_id}/join", json={"nickname": "p2"},
        )

        resp = await client.post(
            f"/games/{game_id}/join", json={"nickname": "p3"},
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_join_not_found(self, client):
        resp = await client.post(
            "/games/nonexistent/join", json={"nickname": "x"},
        )
        assert resp.status_code == 404


class TestStartGame:
    @pytest.mark.asyncio
    async def test_start_success(self, client):
        create_resp = await client.post(
            "/games", json={"player_count": 2},
        )
        game_id = create_resp.json()["game_id"]

        j1 = await client.post(
            f"/games/{game_id}/join", json={"nickname": "p1"},
        )
        token = j1.json()["token"]
        await client.post(
            f"/games/{game_id}/join", json={"nickname": "p2"},
        )

        resp = await client.post(
            f"/games/{game_id}/start", json={"token": token},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "started"

        game_resp = await client.get(f"/games/{game_id}")
        assert game_resp.json()["status"] == "active"

    @pytest.mark.asyncio
    async def test_start_not_enough_players(self, client):
        create_resp = await client.post(
            "/games", json={"player_count": 2},
        )
        game_id = create_resp.json()["game_id"]

        j1 = await client.post(
            f"/games/{game_id}/join", json={"nickname": "p1"},
        )
        token = j1.json()["token"]

        resp = await client.post(
            f"/games/{game_id}/start", json={"token": token},
        )
        assert resp.status_code == 409

    @pytest.mark.asyncio
    async def test_start_wrong_token(self, client):
        create_resp = await client.post(
            "/games", json={"player_count": 2},
        )
        game_id = create_resp.json()["game_id"]

        await client.post(
            f"/games/{game_id}/join", json={"nickname": "p1"},
        )
        j2 = await client.post(
            f"/games/{game_id}/join", json={"nickname": "p2"},
        )
        non_host_token = j2.json()["token"]

        resp = await client.post(
            f"/games/{game_id}/start",
            json={"token": non_host_token},
        )
        assert resp.status_code == 403

    @pytest.mark.asyncio
    async def test_start_not_found(self, client):
        resp = await client.post(
            "/games/nonexistent/start", json={"token": "x"},
        )
        assert resp.status_code == 404


class TestGameStateAfterStart:
    @pytest.mark.asyncio
    async def test_state_includes_engine_data(self, client):
        create_resp = await client.post(
            "/games", json={"player_count": 2},
        )
        game_id = create_resp.json()["game_id"]

        j1 = await client.post(
            f"/games/{game_id}/join", json={"nickname": "p1"},
        )
        token = j1.json()["token"]
        await client.post(
            f"/games/{game_id}/join", json={"nickname": "p2"},
        )
        await client.post(
            f"/games/{game_id}/start", json={"token": token},
        )

        resp = await client.get(f"/games/{game_id}")
        data = resp.json()
        assert "state" in data
        state = data["state"]
        assert "tick" in state
        assert "snakes" in state
        assert len(state["snakes"]) == 2
