"""Tests for the difficulty tier system."""

import numpy as np
import pytest

from smart_snake.ai.agent import DQNAgent
from smart_snake.ai.config import TrainingConfig
from smart_snake.ai.difficulty import (
    ALL_TIERS,
    DEFAULT_WIN_RATE_TARGETS,
    DifficultyAgent,
    DifficultyConfig,
    DifficultyTier,
    TierConfig,
    load_tier_agent,
)
from smart_snake.ai.model_manager import ModelManager
from smart_snake.ai.state import NUM_CHANNELS


def _save_test_checkpoint(path, config=None):
    """Create a minimal checkpoint file for testing."""
    cfg = config or TrainingConfig(
        grid_width=10, grid_height=10,
        conv_channels=(8,), fc_hidden=16,
    )
    agent = DQNAgent(cfg, device="cpu")
    agent.save(path)
    return cfg


class TestDifficultyTier:
    def test_all_tiers_exist(self):
        assert len(ALL_TIERS) == 5
        assert DifficultyTier.BEGINNER in ALL_TIERS
        assert DifficultyTier.IMPOSSIBLE in ALL_TIERS

    def test_tier_values(self):
        assert DifficultyTier.BEGINNER.value == "beginner"
        assert DifficultyTier.HARD.value == "hard"

    def test_default_targets(self):
        for tier in ALL_TIERS:
            assert tier in DEFAULT_WIN_RATE_TARGETS
        assert (
            DEFAULT_WIN_RATE_TARGETS[DifficultyTier.BEGINNER] == 0.0
        )
        assert (
            DEFAULT_WIN_RATE_TARGETS[DifficultyTier.IMPOSSIBLE] == 0.9
        )


class TestDifficultyConfig:
    def test_save_and_load(self, tmp_path):
        config = DifficultyConfig(tiers=[
            TierConfig(
                tier="beginner", checkpoint_path="ckpt/v1.pt",
                win_rate_target=0.0, random_action_prob=0.5,
            ),
            TierConfig(
                tier="hard", checkpoint_path="ckpt/v10.pt",
                win_rate_target=0.75,
            ),
        ])
        path = tmp_path / "tiers.json"
        config.save(path)
        assert path.exists()

        loaded = DifficultyConfig.load(path)
        assert len(loaded.tiers) == 2
        assert loaded.tiers[0].tier == "beginner"
        assert loaded.tiers[0].random_action_prob == 0.5

    def test_get_tier(self):
        config = DifficultyConfig(tiers=[
            TierConfig(
                tier="easy", checkpoint_path="x.pt",
                win_rate_target=0.2,
            ),
        ])
        tier_cfg = config.get_tier("easy")
        assert tier_cfg.checkpoint_path == "x.pt"

    def test_get_tier_by_enum(self):
        config = DifficultyConfig(tiers=[
            TierConfig(
                tier="medium", checkpoint_path="m.pt",
                win_rate_target=0.5,
            ),
        ])
        tier_cfg = config.get_tier(DifficultyTier.MEDIUM)
        assert tier_cfg.checkpoint_path == "m.pt"

    def test_get_tier_not_found(self):
        config = DifficultyConfig(tiers=[])
        with pytest.raises(KeyError, match="impossible"):
            config.get_tier("impossible")


class TestDifficultyAgent:
    def test_select_action(self, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        _save_test_checkpoint(ckpt_path)

        agent = DifficultyAgent(ckpt_path, device="cpu")
        state = np.random.randn(
            NUM_CHANNELS, 10, 10,
        ).astype(np.float32)
        action = agent.select_action(state)
        assert 0 <= action <= 3

    def test_random_action_prob(self, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        _save_test_checkpoint(ckpt_path)

        agent = DifficultyAgent(
            ckpt_path, random_action_prob=1.0, device="cpu",
        )
        state = np.random.randn(
            NUM_CHANNELS, 10, 10,
        ).astype(np.float32)
        actions = {agent.select_action(state) for _ in range(50)}
        assert len(actions) > 1

    def test_select_actions_batch(self, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        _save_test_checkpoint(ckpt_path)

        agent = DifficultyAgent(ckpt_path, device="cpu")
        states = [
            np.random.randn(
                NUM_CHANNELS, 10, 10,
            ).astype(np.float32)
            for _ in range(5)
        ]
        actions = agent.select_actions_batch(states)
        assert len(actions) == 5
        assert all(0 <= a <= 3 for a in actions)

    def test_inference_only_checkpoint(self, tmp_path):
        """Agent can load inference-only exports."""
        ckpt_path = tmp_path / "full.pt"
        _save_test_checkpoint(ckpt_path)

        inf_path = tmp_path / "inference.pt"
        ModelManager.export_for_inference(ckpt_path, inf_path)

        agent = DifficultyAgent(inf_path, device="cpu")
        state = np.random.randn(
            NUM_CHANNELS, 10, 10,
        ).astype(np.float32)
        action = agent.select_action(state)
        assert 0 <= action <= 3


class TestLoadTierAgent:
    def test_load_tier_agent(self, tmp_path):
        ckpt_path = tmp_path / "model.pt"
        _save_test_checkpoint(ckpt_path)

        diff_config = DifficultyConfig(tiers=[
            TierConfig(
                tier="easy", checkpoint_path=str(ckpt_path),
                win_rate_target=0.2, random_action_prob=0.3,
            ),
        ])
        agent = load_tier_agent(diff_config, "easy", device="cpu")
        assert isinstance(agent, DifficultyAgent)
        state = np.random.randn(
            NUM_CHANNELS, 10, 10,
        ).astype(np.float32)
        action = agent.select_action(state)
        assert 0 <= action <= 3
