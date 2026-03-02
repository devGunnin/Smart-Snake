"""Tests for the performance benchmarking utilities."""

import pytest

from smart_snake.ai.benchmark import BenchmarkResult, benchmark_throughput


class TestBenchmarkResult:
    def test_summary_format(self):
        result = BenchmarkResult(
            num_envs=2,
            total_games=10,
            total_steps=500,
            wall_time_seconds=1.5,
            games_per_second=6.67,
            steps_per_second=333.3,
        )
        summary = result.summary()
        assert "2 env(s)" in summary
        assert "10 games" in summary
        assert "games/s" in summary
        assert "steps/s" in summary


class TestBenchmarkThroughput:
    def test_basic_benchmark(self):
        result = benchmark_throughput(
            num_envs=1,
            num_games=5,
            player_count=2,
            grid_width=10,
            grid_height=10,
            max_steps=30,
        )
        assert result.total_games == 5
        assert result.total_steps > 0
        assert result.wall_time_seconds > 0
        assert result.games_per_second > 0
        assert result.steps_per_second > 0

    def test_multi_env_benchmark(self):
        result = benchmark_throughput(
            num_envs=3,
            num_games=6,
            player_count=2,
            grid_width=10,
            grid_height=10,
            max_steps=20,
        )
        assert result.total_games == 6
        assert result.num_envs == 3

    def test_three_player_benchmark(self):
        result = benchmark_throughput(
            num_envs=1,
            num_games=3,
            player_count=3,
            grid_width=10,
            grid_height=10,
            max_steps=20,
        )
        assert result.total_games == 3
        assert result.total_steps > 0

    def test_multi_env_max_steps_applies_per_game(self, monkeypatch):
        class _FakeEnv:
            def __init__(
                self,
                *,
                max_steps: int,
                **_kwargs,
            ):
                self._max_steps = max_steps
                self._steps = 0

            def reset(self, *, seed=None):
                self._steps = 0
                return [], {"seed": seed}

            def step(self, actions):
                _ = actions
                self._steps += 1
                done = self._steps >= self._max_steps
                terminated = [done, done]
                truncated = [False, False]
                return [], [], terminated, truncated, {"game_over": done}

        monkeypatch.setattr(
            "smart_snake.ai.benchmark.MultiSnakeEnv", _FakeEnv,
        )
        result = benchmark_throughput(
            num_envs=3,
            num_games=6,
            player_count=2,
            grid_width=10,
            grid_height=10,
            max_steps=2,
        )
        assert result.total_steps == 12

    def test_num_envs_must_be_positive(self):
        with pytest.raises(ValueError, match="num_envs must be at least 1"):
            benchmark_throughput(num_envs=0, num_games=1)
