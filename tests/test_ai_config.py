"""Tests for the training configuration dataclasses."""

import json

from smart_snake.ai.config import RewardConfig, TrainingConfig


class TestRewardConfig:
    def test_defaults(self):
        cfg = RewardConfig()
        assert cfg.apple == 1.0
        assert cfg.death == -1.0
        assert cfg.step_penalty == -0.01

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

    def test_to_dict(self):
        cfg = TrainingConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["grid_width"] == 20
        assert d["reward"]["apple"] == 1.0

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
