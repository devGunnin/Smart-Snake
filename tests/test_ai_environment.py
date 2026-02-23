"""Tests for Gym-compatible environment wrappers."""

import numpy as np
import pytest

from smart_snake.ai.config import RewardConfig
from smart_snake.ai.environment import (
    NUM_ACTIONS,
    MultiSnakeEnv,
    SnakeEnv,
)
from smart_snake.ai.state import NUM_CHANNELS


class TestSnakeEnvInit:
    def test_spaces(self):
        env = SnakeEnv(width=10, height=10)
        assert env.observation_space.shape == (NUM_CHANNELS, 10, 10)
        assert env.action_space.n == NUM_ACTIONS

    def test_step_before_reset_raises(self):
        env = SnakeEnv(width=10, height=10)
        with pytest.raises(RuntimeError, match="reset"):
            env.step(0)


class TestSnakeEnvReset:
    def test_returns_obs_and_info(self):
        env = SnakeEnv(width=10, height=10, seed=42)
        obs, info = env.reset()
        assert obs.shape == (NUM_CHANNELS, 10, 10)
        assert obs.dtype == np.float32
        assert info["score"] == 0
        assert info["tick"] == 0

    def test_deterministic_with_seed(self):
        env = SnakeEnv(width=10, height=10)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        np.testing.assert_array_equal(obs1, obs2)


class TestSnakeEnvStep:
    def test_step_returns_correct_tuple(self):
        env = SnakeEnv(width=10, height=10, seed=0)
        env.reset()
        obs, reward, terminated, truncated, info = env.step(0)
        assert obs.shape == (NUM_CHANNELS, 10, 10)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "score" in info

    def test_step_penalty_applied(self):
        cfg = RewardConfig(step_penalty=-0.01, apple=1.0, death=-1.0)
        env = SnakeEnv(width=20, height=20, seed=0, reward_config=cfg)
        env.reset()
        _, reward, terminated, _, _ = env.step(0)
        if not terminated:
            assert reward == pytest.approx(-0.01, abs=1e-6)

    def test_death_reward(self):
        cfg = RewardConfig(death=-5.0, step_penalty=0.0)
        env = SnakeEnv(width=5, height=5, seed=0, reward_config=cfg)
        env.reset()
        # Move UP repeatedly until the snake hits a wall.
        total_reward = 0.0
        for _ in range(20):
            _, reward, terminated, _, _ = env.step(0)  # UP
            total_reward += reward
            if terminated:
                break
        assert terminated
        assert -5.0 in [reward]  # last reward should include death penalty

    def test_truncation_at_max_steps(self):
        env = SnakeEnv(width=20, height=20, seed=0, max_steps=5)
        env.reset()
        truncated = False
        for i in range(5):
            action = 3 if i % 2 == 0 else 0  # alternate to avoid wall
            _, _, terminated, truncated, _ = env.step(action)
            if terminated:
                break
        # Should either terminate from death or truncate at step 5.
        assert terminated or truncated

    def test_invalid_action_raises(self):
        env = SnakeEnv(width=10, height=10, seed=0)
        env.reset()
        with pytest.raises(ValueError, match="must be in"):
            env.step(-1)
        with pytest.raises(ValueError, match="must be in"):
            env.step(4)


class TestSnakeEnvRender:
    def test_render_before_reset(self):
        env = SnakeEnv(width=10, height=10)
        assert env.render() == ""

    def test_render_returns_string(self):
        env = SnakeEnv(width=10, height=10, seed=0)
        env.reset()
        text = env.render()
        assert isinstance(text, str)
        lines = text.strip().split("\n")
        assert len(lines) == 10
        assert all(len(line) == 10 for line in lines)


class TestMultiSnakeEnvInit:
    def test_spaces(self):
        env = MultiSnakeEnv(player_count=2, seed=0)
        assert env.observation_space.shape == (NUM_CHANNELS, 20, 20)
        assert env.action_space.n == NUM_ACTIONS
        assert env.num_agents == 2

    def test_step_before_reset_raises(self):
        env = MultiSnakeEnv(player_count=2)
        with pytest.raises(RuntimeError, match="reset"):
            env.step([0, 0])


class TestMultiSnakeEnvReset:
    def test_returns_per_agent_obs(self):
        env = MultiSnakeEnv(player_count=2, seed=0)
        obs_list, info = env.reset()
        assert len(obs_list) == 2
        assert obs_list[0].shape == (NUM_CHANNELS, 20, 20)
        assert info["tick"] == 0

    def test_deterministic_with_seed(self):
        env = MultiSnakeEnv(player_count=2)
        obs1, _ = env.reset(seed=42)
        obs2, _ = env.reset(seed=42)
        for a, b in zip(obs1, obs2, strict=True):
            np.testing.assert_array_equal(a, b)


class TestMultiSnakeEnvStep:
    def test_step_returns_correct_structure(self):
        env = MultiSnakeEnv(player_count=2, seed=0)
        env.reset()
        obs, rewards, terminated, truncated, info = env.step([0, 1])
        assert len(obs) == 2
        assert len(rewards) == 2
        assert len(terminated) == 2
        assert len(truncated) == 2
        assert "game_over" in info

    def test_game_eventually_ends(self):
        env = MultiSnakeEnv(player_count=2, seed=0, max_steps=200)
        env.reset()
        done = False
        for _ in range(200):
            actions = [0, 0]  # both go UP, will eventually hit wall
            _, _, terminated, truncated, info = env.step(actions)
            if all(terminated) or info.get("game_over"):
                done = True
                break
            if any(truncated):
                done = True
                break
        assert done

    def test_three_players(self):
        env = MultiSnakeEnv(player_count=3, seed=0)
        obs, _ = env.reset()
        assert len(obs) == 3
        obs2, rewards, terminated, truncated, info = env.step([0, 1, 2])
        assert len(obs2) == 3
        assert len(rewards) == 3

    def test_invalid_actions_length_raises(self):
        env = MultiSnakeEnv(player_count=2, seed=0)
        env.reset()
        with pytest.raises(ValueError, match="length must match player_count"):
            env.step([0])
        with pytest.raises(ValueError, match="length must match player_count"):
            env.step([0, 1, 2])

    def test_invalid_per_agent_action_raises(self):
        env = MultiSnakeEnv(player_count=2, seed=0)
        env.reset()
        with pytest.raises(ValueError, match="agent 0"):
            env.step([-1, 0])
        with pytest.raises(ValueError, match="agent 1"):
            env.step([0, 4])
