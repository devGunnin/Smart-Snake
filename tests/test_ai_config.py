"""Tests for the training configuration dataclasses."""

import json

import pytest

from smart_snake.ai.config import RewardConfig, TrainingConfig


class TestRewardConfig:
    def test_defaults(self):
        cfg = RewardConfig()
        assert cfg.apple == 3.0
        assert cfg.death == -3.0
        assert cfg.step_penalty == -0.01
        assert cfg.survival_bonus == 0.01
        assert cfg.kill_opponent == 1.0
        assert cfg.apple_approach == 0.1
        assert cfg.apple_retreat == -0.1

    def test_custom_values(self):
        cfg = RewardConfig(apple=2.0, death=-10.0, survival_bonus=0.1)
        assert cfg.apple == 2.0
        assert cfg.death == -10.0
        assert cfg.survival_bonus == 0.1


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.grid_width == 20
        assert cfg.player_count == 2
        assert cfg.dueling is True
        assert cfg.double_dqn is True
        assert cfg.state_encoding == "relative"
        assert cfg.num_envs == 4
        assert cfg.learning_rate == 3e-4
        assert cfg.batch_size == 128
        assert cfg.buffer_size == 500_000
        assert cfg.min_buffer_size == 5_000
        assert cfg.epsilon_decay_steps == 200_000
        assert cfg.target_update_freq == 500
        assert cfg.max_episodes == 50_000
        assert cfg.max_steps_per_episode == 1_000

    def test_to_dict(self):
        cfg = TrainingConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["grid_width"] == 20
        assert d["reward"]["apple"] == 3.0

    def test_save_and_load(self, tmp_path):
        cfg = TrainingConfig(
            grid_width=15, epsilon_start=0.5,
            reward=RewardConfig(apple=3.0),
        )
        path = tmp_path / "config.json"
        cfg.save(path)
        assert path.exists()

        loaded = TrainingConfig.load(path)
        assert loaded.grid_width == 15
        assert loaded.epsilon_start == 0.5
        assert loaded.reward.apple == 3.0
        assert loaded.state_encoding == "relative"

    def test_json_roundtrip(self, tmp_path):
        cfg = TrainingConfig(conv_channels=(16, 32, 64))
        path = tmp_path / "rt.json"
        cfg.save(path)
        loaded = TrainingConfig.load(path)
        assert loaded.conv_channels == (16, 32, 64)

    def test_to_dict_serializable(self):
        cfg = TrainingConfig()
        serialized = json.dumps(cfg.to_dict())
        assert isinstance(serialized, str)

    def test_state_encoding_roundtrip(self, tmp_path):
        cfg = TrainingConfig(state_encoding="relative")
        path = tmp_path / "config_relative.json"
        cfg.save(path)
        loaded = TrainingConfig.load(path)
        assert loaded.state_encoding == "relative"

    def test_invalid_state_encoding_rejected(self):
        with pytest.raises(ValueError, match="state_encoding must be either"):
            TrainingConfig(state_encoding="diagonal")  # type: ignore[arg-type]

    def test_invalid_num_envs_rejected(self):
        with pytest.raises(ValueError, match="num_envs must be at least 1"):
            TrainingConfig(num_envs=0)

    def test_invalid_target_update_freq_rejected(self):
        with pytest.raises(
            ValueError, match="target_update_freq must be at least 1",
        ):
            TrainingConfig(target_update_freq=0)
