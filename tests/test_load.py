"""Load test: concurrent games running simultaneously."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from smart_snake.server import game_manager as gm_mod
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

    @pytest.mark.asyncio
    async def test_rate_limit_stale_entry_removed_on_check(self):
        """Stale IP entries are removed immediately when re-checked."""
        manager = GameManager()
        # Inject a stale entry with an expired timestamp.
        manager._rate_limits["stale-ip"] = [0.0]
        manager._check_rate_limit("stale-ip")
        assert "stale-ip" not in manager._rate_limits

    @pytest.mark.asyncio
    async def test_rate_limit_compaction_sweeps_stale_keys(self):
        """Periodic compaction removes entries for IPs never re-checked."""
        manager = GameManager()
        # Inject stale entries that won't be checked directly.
        manager._rate_limits["old-ip-1"] = [0.0]
        manager._rate_limits["old-ip-2"] = [0.0]
        # Force compaction by backdating last compact time.
        manager._last_rate_compact = 0.0
        with patch.object(
            gm_mod, "_RATE_COMPACT_INTERVAL", 0.0,
        ):
            manager._check_rate_limit("trigger-ip")
        assert "old-ip-1" not in manager._rate_limits
        assert "old-ip-2" not in manager._rate_limits

    @pytest.mark.asyncio
    async def test_rate_limit_compaction_skipped_within_interval(self):
        """Compaction is skipped when called within the interval gate."""
        manager = GameManager()
        manager._rate_limits["old-ip"] = [0.0]
        # Set last compact to current time so interval hasn't elapsed.
        import time
        manager._last_rate_compact = time.monotonic()
        manager._check_rate_limit("other-ip")
        # old-ip was never the checked IP, so it persists (no sweep ran).
        assert "old-ip" in manager._rate_limits

    @pytest.mark.asyncio
    async def test_cleanup_clears_rate_limits(self):
        """GameManager.cleanup() releases rate-limit state."""
        manager = GameManager()
        manager.create_game(client_ip="ip-1")
        manager.create_game(client_ip="ip-2")
        assert len(manager._rate_limits) >= 2
        await manager.cleanup()
        assert len(manager._rate_limits) == 0

    @pytest.mark.asyncio
    async def test_finished_games_pruned_from_registry(self):
        """Keep only a bounded number of finished games in memory."""
        manager = GameManager(max_finished_games=3)
        game_ids: list[str] = []

        for i in range(6):
            game = manager.create_game(
                player_count=2,
                tick_rate_ms=10,
                client_ip=f"prune-{i}",
            )
            game_ids.append(game.game_id)
            host = manager.join_game(game.game_id, f"host-{i}")
            manager.join_game(game.game_id, f"guest-{i}")
            manager.start_game(game.game_id, host.token)

        for _ in range(200):
            await asyncio.sleep(0.05)
            retained = [gid for gid in game_ids if manager.get_game(gid) is not None]
            if len(retained) <= 3:
                break

        retained = [gid for gid in game_ids if manager.get_game(gid) is not None]
        assert len(retained) <= 3
        assert len(retained) < len(game_ids)
        await manager.cleanup()
