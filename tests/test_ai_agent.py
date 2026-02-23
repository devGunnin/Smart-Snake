"""Tests for the DQN agent."""

import numpy as np
import pytest
import torch

from smart_snake.ai.agent import DQNAgent
from smart_snake.ai.config import TrainingConfig
from smart_snake.ai.replay_buffer import Transition
from smart_snake.ai.state import NUM_CHANNELS


def _small_config(**overrides) -> TrainingConfig:
    """Training config with small dimensions for fast tests."""
    defaults = dict(
        grid_width=10,
        grid_height=10,
        player_count=2,
        batch_size=4,
        buffer_size=100,
        min_buffer_size=8,
        target_update_freq=5,
        conv_channels=(8, 16),
        fc_hidden=32,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay_steps=100,
        prioritized_replay=False,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


def _random_transition() -> Transition:
    s = np.random.randn(NUM_CHANNELS, 10, 10).astype(np.float32)
    ns = np.random.randn(NUM_CHANNELS, 10, 10).astype(np.float32)
    return Transition(
        state=s, action=np.random.randint(4), reward=np.random.randn(),
        next_state=ns, done=bool(np.random.randint(2)),
    )


class TestDQNAgentInit:
    def test_creates_networks(self):
        agent = DQNAgent(_small_config(), device="cpu")
        assert agent.online_net is not None
        assert agent.target_net is not None

    def test_dueling_network(self):
        agent = DQNAgent(_small_config(dueling=True), device="cpu")
        from smart_snake.ai.networks import DuelingDQNNetwork
        assert isinstance(agent.online_net, DuelingDQNNetwork)

    def test_standard_network(self):
        agent = DQNAgent(_small_config(dueling=False), device="cpu")
        from smart_snake.ai.networks import DQNNetwork
        assert isinstance(agent.online_net, DQNNetwork)


class TestEpsilon:
    def test_initial_epsilon(self):
        agent = DQNAgent(_small_config(), device="cpu")
        assert agent.epsilon == pytest.approx(1.0)

    def test_epsilon_decays(self):
        agent = DQNAgent(_small_config(), device="cpu")
        agent._step_count = 50
        assert agent.epsilon < 1.0
        assert agent.epsilon > 0.01

    def test_epsilon_at_end(self):
        agent = DQNAgent(_small_config(), device="cpu")
        agent._step_count = 100
        assert agent.epsilon == pytest.approx(0.01)


class TestSelectAction:
    def test_returns_valid_action(self):
        agent = DQNAgent(_small_config(), device="cpu")
        state = np.random.randn(NUM_CHANNELS, 10, 10).astype(np.float32)
        action = agent.select_action(state)
        assert 0 <= action <= 3

    def test_greedy_action_deterministic(self):
        agent = DQNAgent(_small_config(epsilon_start=0.0, epsilon_end=0.0), device="cpu")
        state = np.random.randn(NUM_CHANNELS, 10, 10).astype(np.float32)
        a1 = agent.select_action(state)
        a2 = agent.select_action(state)
        assert a1 == a2

    def test_explores_with_high_epsilon(self):
        agent = DQNAgent(_small_config(epsilon_start=1.0, epsilon_end=1.0), device="cpu")
        state = np.random.randn(NUM_CHANNELS, 10, 10).astype(np.float32)
        # With epsilon=1.0, all actions are random. Over many trials we should
        # see more than one distinct action.
        actions = {agent.select_action(state) for _ in range(50)}
        assert len(actions) > 1


class TestTrainStep:
    def test_can_train_false_when_empty(self):
        agent = DQNAgent(_small_config(), device="cpu")
        assert not agent.can_train()

    def test_can_train_true_when_enough(self):
        agent = DQNAgent(_small_config(), device="cpu")
        for _ in range(8):
            agent.store(_random_transition())
        assert agent.can_train()

    def test_train_step_returns_loss(self):
        agent = DQNAgent(_small_config(), device="cpu")
        for _ in range(10):
            agent.store(_random_transition())
        loss = agent.train_step()
        assert isinstance(loss, float)
        assert loss >= 0.0

    def test_train_step_with_prioritized_replay(self):
        cfg = _small_config(prioritized_replay=True)
        agent = DQNAgent(cfg, device="cpu")
        for _ in range(10):
            agent.store(_random_transition())
        loss = agent.train_step()
        assert isinstance(loss, float)

    def test_target_sync(self):
        agent = DQNAgent(_small_config(target_update_freq=2), device="cpu")
        for _ in range(10):
            agent.store(_random_transition())
        # After 2 train steps the target should be synced.
        agent.train_step()
        agent.train_step()
        for p_on, p_tgt in zip(
            agent.online_net.parameters(), agent.target_net.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(p_on.data, p_tgt.data)

    def test_double_dqn_vs_standard(self):
        cfg_double = _small_config(double_dqn=True)
        cfg_standard = _small_config(double_dqn=False)
        agent_d = DQNAgent(cfg_double, device="cpu")
        agent_s = DQNAgent(cfg_standard, device="cpu")
        for _ in range(10):
            t = _random_transition()
            agent_d.store(t)
            agent_s.store(t)
        # Both should produce a valid loss.
        loss_d = agent_d.train_step()
        loss_s = agent_s.train_step()
        assert isinstance(loss_d, float)
        assert isinstance(loss_s, float)


class TestSaveLoad:
    def test_save_and_load(self, tmp_path):
        agent = DQNAgent(_small_config(), device="cpu")
        for _ in range(10):
            agent.store(_random_transition())
        agent.train_step()

        ckpt = tmp_path / "test_model.pt"
        agent.save(ckpt)
        assert ckpt.exists()

        agent2 = DQNAgent(_small_config(), device="cpu")
        agent2.load(ckpt)

        # Weights should match.
        for p1, p2 in zip(
            agent.online_net.parameters(), agent2.online_net.parameters(),
            strict=True,
        ):
            torch.testing.assert_close(p1.data, p2.data)
        assert agent2._step_count == agent._step_count
