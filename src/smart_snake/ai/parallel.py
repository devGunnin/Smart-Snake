"""Vectorized environment wrappers for parallel training."""

from __future__ import annotations

import logging
import multiprocessing as mp
from typing import Any

import numpy as np

from smart_snake.ai.config import RewardConfig, StateEncodingMode
from smart_snake.ai.environment import MultiSnakeEnv

logger = logging.getLogger(__name__)


class VectorizedEnv:
    """Run *num_envs* :class:`MultiSnakeEnv` instances in a single process.

    Batches ``reset`` and ``step`` across all environments so the agent
    can perform a single forward pass per training step.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        player_count: int = 2,
        grid_width: int | None = None,
        grid_height: int | None = None,
        wall_mode: str = "death",
        max_apples: int = 3,
        initial_snake_length: int = 3,
        reward_config: RewardConfig | None = None,
        max_steps: int = 1000,
        state_encoding: StateEncodingMode = "absolute",
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1.")
        self.num_envs = num_envs
        self._env_kwargs: dict[str, Any] = dict(
            player_count=player_count,
            grid_width=grid_width,
            grid_height=grid_height,
            wall_mode=wall_mode,
            max_apples=max_apples,
            initial_snake_length=initial_snake_length,
            reward_config=reward_config,
            max_steps=max_steps,
            state_encoding=state_encoding,
        )
        self.envs = [
            MultiSnakeEnv(**self._env_kwargs) for _ in range(num_envs)
        ]
        self.num_agents = self.envs[0].num_agents
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

    def reset_all(
        self, *, seeds: list[int] | None = None,
    ) -> tuple[list[list[np.ndarray]], list[dict]]:
        """Reset all environments.

        Returns ``(obs_per_env, info_per_env)`` where each
        ``obs_per_env[i]`` is a list of per-agent observations for
        environment *i*.
        """
        all_obs: list[list[np.ndarray]] = []
        all_info: list[dict] = []
        for i, env in enumerate(self.envs):
            seed = seeds[i] if seeds is not None else None
            obs, info = env.reset(seed=seed)
            all_obs.append(obs)
            all_info.append(info)
        return all_obs, all_info

    def step_all(
        self, actions_per_env: list[list[int]],
    ) -> tuple[
        list[list[np.ndarray]],
        list[list[float]],
        list[list[bool]],
        list[list[bool]],
        list[dict],
    ]:
        """Step all environments in sequence (single-process).

        Args:
            actions_per_env: ``actions_per_env[i]`` is the action list
                for environment *i* (one action per agent).

        Returns:
            ``(obs, rewards, terminated, truncated, infos)`` with each
            entry indexed by ``[env_index]``.
        """
        if len(actions_per_env) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} action lists, "
                f"got {len(actions_per_env)}."
            )
        all_obs: list[list[np.ndarray]] = []
        all_rewards: list[list[float]] = []
        all_terminated: list[list[bool]] = []
        all_truncated: list[list[bool]] = []
        all_info: list[dict] = []
        for env, actions in zip(
            self.envs, actions_per_env, strict=True,
        ):
            obs, rewards, terminated, truncated, info = env.step(actions)
            all_obs.append(obs)
            all_rewards.append(rewards)
            all_terminated.append(terminated)
            all_truncated.append(truncated)
            all_info.append(info)
        return (
            all_obs, all_rewards, all_terminated, all_truncated, all_info,
        )

    def reset_single(
        self, env_idx: int, *, seed: int | None = None,
    ) -> tuple[list[np.ndarray], dict]:
        """Reset a single environment by index."""
        if not 0 <= env_idx < self.num_envs:
            raise IndexError(
                f"env_idx {env_idx} out of range [0, {self.num_envs})."
            )
        return self.envs[env_idx].reset(seed=seed)


def _worker_loop(
    conn: mp.connection.Connection,
    env_kwargs: dict[str, Any],
) -> None:
    """Worker process main loop for :class:`SubprocessVectorizedEnv`."""
    env = MultiSnakeEnv(**env_kwargs)
    while True:
        cmd, data = conn.recv()
        if cmd == "reset":
            result = env.reset(seed=data)
            conn.send(result)
        elif cmd == "step":
            result = env.step(data)
            conn.send(result)
        elif cmd == "close":
            conn.close()
            break


class SubprocessVectorizedEnv:
    """Run environments in separate worker processes.

    Useful for CPU-bound simulation when in-process vectorization
    is bottlenecked by the GIL.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        player_count: int = 2,
        grid_width: int | None = None,
        grid_height: int | None = None,
        wall_mode: str = "death",
        max_apples: int = 3,
        initial_snake_length: int = 3,
        reward_config: RewardConfig | None = None,
        max_steps: int = 1000,
        state_encoding: StateEncodingMode = "absolute",
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be at least 1.")
        self.num_envs = num_envs
        env_kwargs: dict[str, Any] = dict(
            player_count=player_count,
            grid_width=grid_width,
            grid_height=grid_height,
            wall_mode=wall_mode,
            max_apples=max_apples,
            initial_snake_length=initial_snake_length,
            max_steps=max_steps,
            state_encoding=state_encoding,
        )
        if reward_config is not None:
            env_kwargs["reward_config"] = reward_config

        probe = MultiSnakeEnv(**env_kwargs)
        self.num_agents = probe.num_agents
        self.observation_space = probe.observation_space
        self.action_space = probe.action_space

        ctx = mp.get_context("spawn")
        self._parent_conns: list[mp.connection.Connection] = []
        self._procs: list[mp.Process] = []
        for _ in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            proc = ctx.Process(
                target=_worker_loop,
                args=(child_conn, env_kwargs),
                daemon=True,
            )
            proc.start()
            child_conn.close()
            self._parent_conns.append(parent_conn)
            self._procs.append(proc)

    def reset_all(
        self, *, seeds: list[int] | None = None,
    ) -> tuple[list[list[np.ndarray]], list[dict]]:
        """Reset all worker environments."""
        for i, conn in enumerate(self._parent_conns):
            seed = seeds[i] if seeds is not None else None
            conn.send(("reset", seed))
        all_obs: list[list[np.ndarray]] = []
        all_info: list[dict] = []
        for conn in self._parent_conns:
            obs, info = conn.recv()
            all_obs.append(obs)
            all_info.append(info)
        return all_obs, all_info

    def step_all(
        self, actions_per_env: list[list[int]],
    ) -> tuple[
        list[list[np.ndarray]],
        list[list[float]],
        list[list[bool]],
        list[list[bool]],
        list[dict],
    ]:
        """Step all worker environments in parallel."""
        if len(actions_per_env) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} action lists, "
                f"got {len(actions_per_env)}."
            )
        for conn, actions in zip(
            self._parent_conns, actions_per_env, strict=True,
        ):
            conn.send(("step", actions))
        all_obs: list[list[np.ndarray]] = []
        all_rewards: list[list[float]] = []
        all_terminated: list[list[bool]] = []
        all_truncated: list[list[bool]] = []
        all_info: list[dict] = []
        for conn in self._parent_conns:
            obs, rewards, terminated, truncated, info = conn.recv()
            all_obs.append(obs)
            all_rewards.append(rewards)
            all_terminated.append(terminated)
            all_truncated.append(truncated)
            all_info.append(info)
        return (
            all_obs, all_rewards, all_terminated, all_truncated, all_info,
        )

    def reset_single(
        self, env_idx: int, *, seed: int | None = None,
    ) -> tuple[list[np.ndarray], dict]:
        """Reset a single worker environment."""
        if not 0 <= env_idx < self.num_envs:
            raise IndexError(
                f"env_idx {env_idx} out of range [0, {self.num_envs})."
            )
        self._parent_conns[env_idx].send(("reset", seed))
        return self._parent_conns[env_idx].recv()

    def close(self) -> None:
        """Shut down all worker processes."""
        for conn in self._parent_conns:
            try:
                conn.send(("close", None))
                conn.close()
            except (BrokenPipeError, OSError):
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()

    def __del__(self) -> None:
        self.close()
