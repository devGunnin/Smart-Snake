"""Performance benchmarking utilities for training throughput."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from smart_snake.ai.environment import MultiSnakeEnv

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Results from a throughput benchmark run."""

    num_envs: int
    total_games: int
    total_steps: int
    wall_time_seconds: float
    games_per_second: float
    steps_per_second: float

    def summary(self) -> str:
        return (
            f"Benchmark: {self.num_envs} env(s), "
            f"{self.total_games} games, {self.total_steps} steps in "
            f"{self.wall_time_seconds:.2f}s | "
            f"{self.games_per_second:.1f} games/s, "
            f"{self.steps_per_second:.1f} steps/s"
        )


def benchmark_throughput(
    *,
    num_envs: int = 1,
    num_games: int = 100,
    player_count: int = 2,
    grid_width: int = 20,
    grid_height: int = 20,
    max_steps: int = 200,
) -> BenchmarkResult:
    """Measure raw simulation throughput (no neural network).

    Runs *num_games* episodes across *num_envs* environments using
    random actions and reports games/second and steps/second.
    """
    envs = [
        MultiSnakeEnv(
            player_count=player_count,
            grid_width=grid_width,
            grid_height=grid_height,
            max_steps=max_steps,
        )
        for _ in range(num_envs)
    ]
    rng = np.random.default_rng(42)

    total_steps = 0
    total_games = 0
    start = time.perf_counter()

    while total_games < num_games:
        batch_size = min(num_envs, num_games - total_games)
        active_envs = envs[:batch_size]

        for env in active_envs:
            env.reset(seed=int(rng.integers(2**31)))

        dones = [False] * batch_size
        step_count = 0

        while not all(dones) and step_count < max_steps:
            for i, env in enumerate(active_envs):
                if dones[i]:
                    continue
                actions = rng.integers(4, size=player_count).tolist()
                _, _, terminated, truncated, info = env.step(actions)
                total_steps += 1
                step_count += 1
                if info.get("game_over") or all(
                    t or tr
                    for t, tr in zip(terminated, truncated, strict=True)
                ):
                    dones[i] = True

        total_games += batch_size

    elapsed = time.perf_counter() - start
    result = BenchmarkResult(
        num_envs=num_envs,
        total_games=total_games,
        total_steps=total_steps,
        wall_time_seconds=elapsed,
        games_per_second=total_games / max(elapsed, 1e-9),
        steps_per_second=total_steps / max(elapsed, 1e-9),
    )
    logger.info(result.summary())
    return result
