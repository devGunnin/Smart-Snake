"""WebSocket integration tests for real-time gameplay."""

from __future__ import annotations

import json

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from smart_snake.server.app import create_app
from smart_snake.server.game_manager import GameManager


@pytest.fixture()
def tc():
    """Starlette sync TestClient — shares a single event loop for both
    REST calls and WebSocket connections."""
    application = create_app()
    application.state.game_manager = GameManager()
    return TestClient(application)


def _create_start_game(tc, player_count=2):
    """Create a game, join all players, start it, return (game_id, tokens)."""
    resp = tc.post(
        "/games",
        json={"player_count": player_count, "tick_rate_ms": 50},
    )
    assert resp.status_code == 201
    game_id = resp.json()["game_id"]
    tokens = []
    for i in range(player_count):
        j = tc.post(
            f"/games/{game_id}/join", json={"nickname": f"p{i}"},
        )
        assert j.status_code == 201
        tokens.append(j.json()["token"])
    tc.post(f"/games/{game_id}/start", json={"token": tokens[0]})
    return game_id, tokens


class TestPlayWebSocket:
    def test_connect_and_receive_initial_state(self, tc):
        game_id, tokens = _create_start_game(tc)

        with tc.websocket_connect(
            f"/games/{game_id}/play?token={tokens[0]}",
        ) as ws:
            raw = ws.receive_text()
            state = json.loads(raw)
            assert "tick" in state
            assert "snakes" in state
            assert "grid" in state
            assert len(state["snakes"]) == 2

    def test_send_direction_accepted(self, tc):
        game_id, tokens = _create_start_game(tc)

        with tc.websocket_connect(
            f"/games/{game_id}/play?token={tokens[0]}",
        ) as ws:
            # Consume initial state.
            ws.receive_text()
            # Send a direction change — should not error.
            ws.send_text(json.dumps({"direction": "up"}))

    def test_invalid_token_rejected(self, tc):
        game_id, tokens = _create_start_game(tc)

        with pytest.raises(WebSocketDisconnect), tc.websocket_connect(
            f"/games/{game_id}/play?token=badtoken",
        ):
            pass

    def test_nonexistent_game_rejected(self, tc):
        with pytest.raises(WebSocketDisconnect), tc.websocket_connect(
            "/games/nonexistent/play?token=x",
        ):
            pass

    def test_two_players_both_receive_initial_state(self, tc):
        game_id, tokens = _create_start_game(tc)

        with tc.websocket_connect(
            f"/games/{game_id}/play?token={tokens[0]}",
        ) as ws0, tc.websocket_connect(
            f"/games/{game_id}/play?token={tokens[1]}",
        ) as ws1:
            state0 = json.loads(ws0.receive_text())
            state1 = json.loads(ws1.receive_text())
            assert "tick" in state0
            assert "tick" in state1
            assert len(state0["snakes"]) == 2
            assert len(state1["snakes"]) == 2


class TestSpectateWebSocket:
    def test_spectator_receives_initial_state(self, tc):
        game_id, tokens = _create_start_game(tc)

        with tc.websocket_connect(
            f"/games/{game_id}/spectate",
        ) as ws:
            raw = ws.receive_text()
            state = json.loads(raw)
            assert "tick" in state
            assert "snakes" in state

    def test_spectate_nonexistent_game(self, tc):
        with pytest.raises(WebSocketDisconnect), tc.websocket_connect(
            "/games/nonexistent/spectate",
        ):
            pass


class TestDisconnectHandling:
    def test_player_disconnect_game_continues(self, tc):
        game_id, tokens = _create_start_game(tc)

        with tc.websocket_connect(
            f"/games/{game_id}/play?token={tokens[0]}",
        ) as ws:
            ws.receive_text()

        # Game should still be running after player disconnects.
        resp = tc.get(f"/games/{game_id}")
        assert resp.json()["status"] in ("active", "finished")

    def test_invalid_json_ignored(self, tc):
        game_id, tokens = _create_start_game(tc)

        with tc.websocket_connect(
            f"/games/{game_id}/play?token={tokens[0]}",
        ) as ws:
            # Consume initial state.
            ws.receive_text()
            # Send garbage — should be silently ignored.
            ws.send_text("not-json")
            ws.send_text("[]")
            ws.send_text("123")
            ws.send_text(json.dumps({"direction": "invalid_dir"}))
            ws.send_text(json.dumps({"no_direction_key": True}))

    def test_reconnect_does_not_drop_active_replacement_socket(self, tc):
        game_id, tokens = _create_start_game(tc)
        game = tc.app.state.game_manager.get_game(game_id)
        slot = game.players[tokens[0]]

        ws1_ctx = tc.websocket_connect(f"/games/{game_id}/play?token={tokens[0]}")
        ws1_ctx.__enter__()
        ws1_closed = False
        ws2_ctx = None
        try:
            ws1_ctx.receive_text()
            ws2_ctx = tc.websocket_connect(f"/games/{game_id}/play?token={tokens[0]}")
            ws2_ctx.__enter__()
            ws2_ctx.receive_text()

            # Old connection drops after a newer one is established.
            ws1_ctx.__exit__(None, None, None)
            ws1_closed = True
            assert slot.websocket is not None
            assert slot.connected is True
        finally:
            if ws2_ctx is not None:
                ws2_ctx.__exit__(None, None, None)
            if not ws1_closed:
                ws1_ctx.__exit__(None, None, None)
