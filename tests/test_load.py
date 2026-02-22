"""Load test: concurrent games running simultaneously."""

from __future__ import annotations

import asyncio

import pytest

from smart_snake.server.game_manager import GameManager


class TestConcurrentGames:
    @pytest.mark.asyncio
    async def test_50_concurrent_games(self):
        """Spin up 50 games with 2 players each; verify all finish."""
        manager = GameManager()
        game_ids: list[str] = []
        host_tokens: list[str] = []

        for i in range(50):
            game = manager.create_game(
                player_count=2,
                tick_rate_ms=10,
                client_ip=f"test-{i % 10}",
            )
            game_ids.append(game.game_id)

            slot0 = manager.join_game(game.game_id, f"p0-{i}")
            manager.join_game(game.game_id, f"p1-{i}")
            host_tokens.append(slot0.token)

        for gid, token in zip(game_ids, host_tokens, strict=True):
            manager.start_game(gid, token)

        # Wait for games to finish (snakes collide into walls).
        for _ in range(200):
            await asyncio.sleep(0.05)
            statuses = [
                manager.get_game(gid).status.value for gid in game_ids
            ]
            if all(s == "finished" for s in statuses):
                break

        finished = sum(
            1 for gid in game_ids
            if manager.get_game(gid).status.value == "finished"
        )
        assert finished == 50, f"Only {finished}/50 games finished"
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_mixed_player_counts(self):
        """Run 20 games with mixed 2/3/4 player counts."""
        manager = GameManager()
        game_ids: list[str] = []
        host_tokens: list[str] = []

        player_counts = [2, 3, 4] * 7
        for i, pc in enumerate(player_counts[:20]):
            game = manager.create_game(
                player_count=pc,
                tick_rate_ms=10,
                client_ip=f"test-{i % 10}",
            )
            game_ids.append(game.game_id)

            tokens = []
            for j in range(pc):
                slot = manager.join_game(game.game_id, f"p{j}-g{i}")
                tokens.append(slot.token)
            host_tokens.append(tokens[0])

        for gid, token in zip(game_ids, host_tokens, strict=True):
            manager.start_game(gid, token)

        for _ in range(200):
            await asyncio.sleep(0.05)
            statuses = [
                manager.get_game(gid).status.value for gid in game_ids
            ]
            if all(s == "finished" for s in statuses):
                break

        finished = sum(
            1 for gid in game_ids
            if manager.get_game(gid).status.value == "finished"
        )
        assert finished == 20, f"Only {finished}/20 games finished"
        await manager.cleanup()

    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        """Verify rate limiter blocks excessive creation from one IP."""
        manager = GameManager()
        for _ in range(10):
            manager.create_game(client_ip="same-ip")

        with pytest.raises(ValueError, match="Rate limit"):
            manager.create_game(client_ip="same-ip")

        # Different IP should still work.
        manager.create_game(client_ip="other-ip")
