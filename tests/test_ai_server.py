"""Tests for server-side AI opponent integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
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


class TestCreateGameWithAi:
    @pytest.mark.asyncio
    async def test_create_with_one_ai(self, client):
        resp = await client.post("/games", json={
            "player_count": 2,
            "ai_opponents": [{"difficulty": "medium"}],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["ai_count"] == 1
        assert data["player_count"] == 1

    @pytest.mark.asyncio
    async def test_create_with_multiple_ai(self, client):
        resp = await client.post("/games", json={
            "player_count": 4,
            "ai_opponents": [
                {"difficulty": "easy"},
                {"difficulty": "hard"},
                {"difficulty": "impossible"},
            ],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["ai_count"] == 3
        assert data["player_count"] == 3

    @pytest.mark.asyncio
    async def test_ai_too_many_rejects(self, client):
        resp = await client.post("/games", json={
            "player_count": 2,
            "ai_opponents": [
                {"difficulty": "easy"},
                {"difficulty": "hard"},
            ],
        })
        assert resp.status_code == 422
        assert "human" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_ai_invalid_difficulty(self, client):
        resp = await client.post("/games", json={
            "player_count": 2,
            "ai_opponents": [{"difficulty": "invalid"}],
        })
        assert resp.status_code == 422
        assert "invalid" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_ai_default_difficulty(self, client):
        resp = await client.post("/games", json={
            "player_count": 2,
            "ai_opponents": [{}],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["ai_count"] == 1

    @pytest.mark.asyncio
    async def test_no_ai_backwards_compatible(self, client):
        resp = await client.post("/games", json={})
        assert resp.status_code == 201
        data = resp.json()
        assert data["ai_count"] == 0
        assert data["player_count"] == 0


class TestGameDetailWithAi:
    @pytest.mark.asyncio
    async def test_ai_players_in_detail(self, client):
        create_resp = await client.post("/games", json={
            "player_count": 3,
            "ai_opponents": [{"difficulty": "hard"}],
        })
        game_id = create_resp.json()["game_id"]
        resp = await client.get(f"/games/{game_id}")
        data = resp.json()

        assert data["ai_count"] == 1
        ai_players = [p for p in data["players"] if p["is_ai"]]
        assert len(ai_players) == 1
        assert "AI" in ai_players[0]["nickname"]
        assert "Hard" in ai_players[0]["nickname"]

    @pytest.mark.asyncio
    async def test_human_joins_after_ai(self, client):
        create_resp = await client.post("/games", json={
            "player_count": 2,
            "ai_opponents": [{"difficulty": "easy"}],
        })
        game_id = create_resp.json()["game_id"]

        join_resp = await client.post(
            f"/games/{game_id}/join", json={"nickname": "human"},
        )
        assert join_resp.status_code == 201
        data = join_resp.json()
        assert data["snake_id"] == 1
        assert data["nickname"] == "human"

    @pytest.mark.asyncio
    async def test_lobby_full_with_ai_and_humans(self, client):
        create_resp = await client.post("/games", json={
            "player_count": 2,
            "ai_opponents": [{"difficulty": "medium"}],
        })
        game_id = create_resp.json()["game_id"]

        await client.post(
            f"/games/{game_id}/join", json={"nickname": "p1"},
        )
        resp = await client.post(
            f"/games/{game_id}/join", json={"nickname": "p2"},
        )
        assert resp.status_code == 409


class TestGameListWithAi:
    @pytest.mark.asyncio
    async def test_list_includes_ai_count(self, client):
        await client.post("/games", json={
            "player_count": 3,
            "ai_opponents": [
                {"difficulty": "easy"},
                {"difficulty": "hard"},
            ],
        })
        resp = await client.get("/games")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["ai_count"] == 2


class TestGameManagerAiDirect:
    def test_ai_slot_properties(self):
        manager = GameManager()
        game = manager.create_game(
            player_count=3,
            ai_opponents=[
                {"difficulty": "beginner"},
                {"difficulty": "impossible"},
            ],
        )
        ai_slots = [s for s in game.players.values() if s.is_ai]
        assert len(ai_slots) == 2
        assert all(s.token.startswith("ai-") for s in ai_slots)
        assert ai_slots[0].snake_id == 0
        assert ai_slots[1].snake_id == 1

    def test_host_not_set_for_ai_only(self):
        manager = GameManager()
        game = manager.create_game(
            player_count=2,
            ai_opponents=[{"difficulty": "medium"}],
        )
        assert game.host_token is None

    def test_human_becomes_host(self):
        manager = GameManager()
        game = manager.create_game(
            player_count=2,
            ai_opponents=[{"difficulty": "medium"}],
        )
        slot = manager.join_game(game.game_id, "human")
        assert game.host_token == slot.token
        assert not slot.is_ai


class TestAiAgentLoading:
    def test_start_without_checkpoint_fails(self):
        manager = GameManager(checkpoint_dir="/nonexistent")
        game = manager.create_game(
            player_count=2,
            ai_opponents=[{"difficulty": "medium"}],
        )
        slot = manager.join_game(game.game_id, "human")
        with pytest.raises(ValueError, match="checkpoint"):
            manager.start_game(game.game_id, slot.token)

    def test_set_ai_directions_with_mock_agent(self):
        manager = GameManager()
        game = manager.create_game(
            player_count=2,
            ai_opponents=[{"difficulty": "medium"}],
        )
        manager.join_game(game.game_id, "human")

        from smart_snake.multiplayer import MultiplayerEngine

        game.engine = MultiplayerEngine(game.config)
        game.status = "active"

        mock_agent = MagicMock()
        mock_agent.select_action.return_value = 0
        game.ai_agents[0] = mock_agent

        manager._set_ai_directions(game)
        mock_agent.select_action.assert_called_once()
        call_arg = mock_agent.select_action.call_args[0][0]
        assert isinstance(call_arg, np.ndarray)

    def test_set_ai_directions_skips_dead_snake(self):
        manager = GameManager()
        game = manager.create_game(
            player_count=2,
            ai_opponents=[{"difficulty": "medium"}],
        )
        manager.join_game(game.game_id, "human")  # fill lobby

        from smart_snake.multiplayer import MultiplayerEngine

        game.engine = MultiplayerEngine(game.config)
        game.engine.snakes[0].alive = False

        mock_agent = MagicMock()
        game.ai_agents[0] = mock_agent

        manager._set_ai_directions(game)
        mock_agent.select_action.assert_not_called()

    def test_set_ai_directions_noop_without_agents(self):
        manager = GameManager()
        game = manager.create_game(player_count=2)
        manager.join_game(game.game_id, "p1")
        manager.join_game(game.game_id, "p2")

        from smart_snake.multiplayer import MultiplayerEngine

        game.engine = MultiplayerEngine(game.config)
        manager._set_ai_directions(game)
