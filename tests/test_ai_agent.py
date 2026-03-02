"""Tests for the MAPPO agent."""

import numpy as np
import torch

from smart_snake.ai.agent import MAPPOAgent
from smart_snake.ai.config import TrainingConfig
from smart_snake.ai.networks import ActorCriticNetwork
from smart_snake.ai.state import NUM_CHANNELS


def _small_config(**overrides) -> TrainingConfig:
    """Training config with small dimensions for fast tests."""
    defaults = dict(
        grid_width=10,
        grid_height=10,
        player_count=2,
        conv_channels=(8, 16),
        fc_hidden=32,
        clip_ratio=0.2,
        gae_lambda=0.95,
        entropy_coeff=0.01,
        ppo_epochs=2,
        num_minibatches=2,
        rollout_steps=16,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _random_batch(batch_size: int = 8) -> dict[str, np.ndarray]:
    """Create a random mini-batch for PPO update testing."""
    return {
        "states": np.random.randn(
            batch_size, NUM_CHANNELS, 10, 10,
        ).astype(np.float32),
        "actions": np.random.randint(0, 4, size=batch_size).astype(
            np.int64,
        ),
        "log_probs": np.random.randn(batch_size).astype(np.float32),
        "advantages": np.random.randn(batch_size).astype(np.float32),
        "returns": np.random.randn(batch_size).astype(np.float32),
        "action_masks": np.ones(
            (batch_size, 4), dtype=bool,
        ),
    }


class TestMAPPOAgentInit:
    def test_creates_network(self):
        agent = MAPPOAgent(_small_config(), device="cpu")
        assert isinstance(agent.network, ActorCriticNetwork)

    def test_initial_step_count(self):
        agent = MAPPOAgent(_small_config(), device="cpu")
        assert agent._step_count == 0


class TestSelectAction:
    def test_returns_valid_action(self):
        agent = MAPPOAgent(_small_config(), device="cpu")
        state = np.random.randn(
            NUM_CHANNELS, 10, 10,
        ).astype(np.float32)
        action, log_prob, value = agent.select_action(state)
        assert 0 <= action <= 3
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_action_masking(self):
        agent = MAPPOAgent(_small_config(), device="cpu")
        state = np.random.randn(
            NUM_CHANNELS, 10, 10,
        ).astype(np.float32)
        # Mask out action 0 and 1.
        mask = np.array([False, False, True, True])
        actions = set()
        for _ in range(50):
            a, _, _ = agent.select_action(state, action_mask=mask)
            actions.add(a)
        assert all(a in {2, 3} for a in actions)

    def test_batch_returns_correct_length(self):
        agent = MAPPOAgent(_small_config(), device="cpu")
        states = [
            np.random.randn(
                NUM_CHANNELS, 10, 10,
            ).astype(np.float32)
            for _ in range(5)
        ]
        actions, log_probs, values = (
            agent.select_actions_batch(states)
        )
        assert len(actions) == 5
        assert len(log_probs) == 5
        assert len(values) == 5
        assert all(0 <= a <= 3 for a in actions)

    def test_batch_with_masks(self):
        agent = MAPPOAgent(_small_config(), device="cpu")
        states = [
            np.random.randn(
                NUM_CHANNELS, 10, 10,
            ).astype(np.float32)
            for _ in range(4)
        ]
        # Only allow action 2 for all.
        masks = [np.array([False, False, True, False])] * 4
        actions, _, _ = agent.select_actions_batch(
            states, action_masks=masks,
        )
        assert all(a == 2 for a in actions)


class TestGetValues:
    def test_returns_values(self):
        agent = MAPPOAgent(_small_config(), device="cpu")
        states = [
            np.random.randn(
                NUM_CHANNELS, 10, 10,
            ).astype(np.float32)
            for _ in range(3)
        ]
        values = agent.get_values(states)
        assert len(values) == 3
        assert all(isinstance(v, float) for v in values)


class TestUpdate:
    def test_update_returns_metrics(self):
        agent = MAPPOAgent(_small_config(), device="cpu")
        batches = [_random_batch(), _random_batch()]
        metrics = agent.update(batches)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "total_loss" in metrics
        assert "clip_fraction" in metrics

    def test_step_count_increases(self):
        agent = MAPPOAgent(_small_config(), device="cpu")
        batches = [_random_batch()]
        agent.update(batches)
        assert agent._step_count == 1
        agent.update(batches)
        assert agent._step_count == 2

    def test_action_masking_in_update(self):
        agent = MAPPOAgent(_small_config(), device="cpu")
        batch = _random_batch()
        # Mask out one action per sample.
        batch["action_masks"][:, 0] = False
        metrics = agent.update([batch])
        assert isinstance(metrics["policy_loss"], float)


class TestSnapshotPool:
    def test_save_snapshot(self):
        agent = MAPPOAgent(_small_config(), device="cpu")
        assert agent.snapshot_pool_size == 0
        agent.save_snapshot()
        assert agent.snapshot_pool_size == 1

    def test_pool_size_limit(self):
        agent = MAPPOAgent(
            _small_config(snapshot_pool_size=3), device="cpu",
        )
        for _ in range(5):
            agent.save_snapshot()
        assert agent.snapshot_pool_size == 3

    def test_sample_opponent_empty_pool(self):
        agent = MAPPOAgent(_small_config(), device="cpu")
        opp = agent.sample_opponent()
        assert opp is None

    def test_sample_opponent_returns_network(self):
        agent = MAPPOAgent(
            _small_config(latest_vs_latest_prob=0.0),
            device="cpu",
        )
        agent.save_snapshot()
        opp = agent.sample_opponent()
        assert isinstance(opp, ActorCriticNetwork)


class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        agent = MAPPOAgent(_small_config(), device="cpu")
        agent.update([_random_batch()])

        ckpt = tmp_path / "test_model.pt"
        agent.save(ckpt)
        assert ckpt.exists()

        agent2 = MAPPOAgent(_small_config(), device="cpu")
        agent2.load(ckpt)

        # Weights should match.
        for p1, p2 in zip(
            agent.network.parameters(),
            agent2.network.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(p1.data, p2.data)
        assert agent2._step_count == agent._step_count
