"""Tests for vectorized parallel environments."""

import numpy as np
import pytest

from smart_snake.ai.parallel import VectorizedEnv


class TestVectorizedEnv:
    def test_init(self):
        venv = VectorizedEnv(
            3, player_count=2, grid_width=10, grid_height=10,
        )
        assert venv.num_envs == 3
        assert venv.num_agents == 2
        assert len(venv.envs) == 3

    def test_invalid_num_envs(self):
        with pytest.raises(ValueError, match="num_envs must be at least 1"):
            VectorizedEnv(
                0, player_count=2, grid_width=10, grid_height=10,
            )

    def test_reset_all(self):
        venv = VectorizedEnv(
            2, player_count=2, grid_width=10, grid_height=10,
        )
        obs, infos = venv.reset_all(seeds=[42, 43])
        assert len(obs) == 2
        assert len(infos) == 2
        for env_obs in obs:
            assert len(env_obs) == 2
            assert env_obs[0].shape == venv.observation_space.shape

    def test_reset_all_no_seeds(self):
        venv = VectorizedEnv(
            2, player_count=2, grid_width=10, grid_height=10,
        )
        obs, infos = venv.reset_all()
        assert len(obs) == 2

    def test_step_all(self):
        venv = VectorizedEnv(
            2, player_count=2, grid_width=10, grid_height=10,
        )
        venv.reset_all(seeds=[42, 43])
        actions = [[0, 1], [2, 3]]
        obs, rewards, terminated, truncated, infos = (
            venv.step_all(actions)
        )
        assert len(obs) == 2
        assert len(rewards) == 2
        assert len(terminated) == 2
        assert len(truncated) == 2
        assert len(infos) == 2
        for env_rewards in rewards:
            assert len(env_rewards) == 2

    def test_step_all_wrong_count(self):
        venv = VectorizedEnv(
            2, player_count=2, grid_width=10, grid_height=10,
        )
        venv.reset_all()
        with pytest.raises(ValueError, match="Expected 2 action lists"):
            venv.step_all([[0, 1]])

    def test_reset_single(self):
        venv = VectorizedEnv(
            3, player_count=2, grid_width=10, grid_height=10,
        )
        venv.reset_all()
        obs, info = venv.reset_single(1, seed=99)
        assert len(obs) == 2
        assert isinstance(info, dict)

    def test_reset_single_out_of_range(self):
        venv = VectorizedEnv(
            2, player_count=2, grid_width=10, grid_height=10,
        )
        with pytest.raises(IndexError, match="out of range"):
            venv.reset_single(5)

    def test_multiple_steps(self):
        venv = VectorizedEnv(
            2, player_count=2, grid_width=10, grid_height=10,
        )
        venv.reset_all(seeds=[10, 20])
        rng = np.random.default_rng(0)
        for _ in range(5):
            actions = [
                rng.integers(4, size=2).tolist() for _ in range(2)
            ]
            obs, rewards, terminated, truncated, infos = (
                venv.step_all(actions)
            )
            assert len(obs) == 2

    def test_observation_space_matches_envs(self):
        venv = VectorizedEnv(
            2, player_count=2, grid_width=15, grid_height=15,
        )
        assert (
            venv.observation_space.shape
            == venv.envs[0].observation_space.shape
        )
        assert venv.action_space.n == venv.envs[0].action_space.n
