"""Tests for the on-policy rollout buffer."""

import numpy as np
import pytest

from smart_snake.ai.replay_buffer import RolloutBuffer
from smart_snake.ai.state import NUM_CHANNELS


class TestRolloutBufferInit:
    def test_init(self):
        buf = RolloutBuffer(
            rollout_steps=10, num_agents=2,
            obs_shape=(NUM_CHANNELS, 10, 10), num_actions=4,
        )
        assert buf.rollout_steps == 10
        assert buf.num_agents == 2
        assert len(buf) == 0
        assert not buf.full

    def test_invalid_rollout_steps(self):
        with pytest.raises(ValueError, match="rollout_steps"):
            RolloutBuffer(
                rollout_steps=0, num_agents=2,
                obs_shape=(NUM_CHANNELS, 10, 10), num_actions=4,
            )

    def test_invalid_num_agents(self):
        with pytest.raises(ValueError, match="num_agents"):
            RolloutBuffer(
                rollout_steps=10, num_agents=0,
                obs_shape=(NUM_CHANNELS, 10, 10), num_actions=4,
            )


class TestRolloutBufferAdd:
    def _make_step_data(self, num_agents: int = 2):
        return dict(
            states=np.random.randn(
                num_agents, NUM_CHANNELS, 10, 10,
            ).astype(np.float32),
            actions=np.random.randint(
                0, 4, size=num_agents,
            ).astype(np.int64),
            rewards=np.random.randn(num_agents).astype(np.float32),
            values=np.random.randn(num_agents).astype(np.float32),
            log_probs=np.random.randn(num_agents).astype(np.float32),
            dones=np.zeros(num_agents, dtype=np.float32),
            action_masks=np.ones(
                (num_agents, 4), dtype=bool,
            ),
        )

    def test_add_increments_step(self):
        buf = RolloutBuffer(
            rollout_steps=5, num_agents=2,
            obs_shape=(NUM_CHANNELS, 10, 10), num_actions=4,
        )
        buf.add(**self._make_step_data())
        assert len(buf) == 1
        assert not buf.full

    def test_full_after_rollout_steps(self):
        buf = RolloutBuffer(
            rollout_steps=3, num_agents=2,
            obs_shape=(NUM_CHANNELS, 10, 10), num_actions=4,
        )
        for _ in range(3):
            buf.add(**self._make_step_data())
        assert buf.full
        assert len(buf) == 3

    def test_add_when_full_raises(self):
        buf = RolloutBuffer(
            rollout_steps=2, num_agents=2,
            obs_shape=(NUM_CHANNELS, 10, 10), num_actions=4,
        )
        for _ in range(2):
            buf.add(**self._make_step_data())
        with pytest.raises(RuntimeError, match="full"):
            buf.add(**self._make_step_data())


class TestRolloutBufferComputeReturns:
    def test_compute_returns(self):
        buf = RolloutBuffer(
            rollout_steps=4, num_agents=2,
            obs_shape=(NUM_CHANNELS, 10, 10), num_actions=4,
        )
        for _ in range(4):
            buf.add(
                states=np.zeros(
                    (2, NUM_CHANNELS, 10, 10), dtype=np.float32,
                ),
                actions=np.array([0, 1], dtype=np.int64),
                rewards=np.array(
                    [1.0, 0.5], dtype=np.float32,
                ),
                values=np.array(
                    [0.5, 0.3], dtype=np.float32,
                ),
                log_probs=np.array(
                    [-0.5, -0.7], dtype=np.float32,
                ),
                dones=np.zeros(2, dtype=np.float32),
                action_masks=np.ones((2, 4), dtype=bool),
            )
        last_values = np.array([0.5, 0.3], dtype=np.float32)
        buf.compute_returns(last_values, gamma=0.99, gae_lambda=0.95)

        # Advantages should be computed.
        assert buf.advantages[:4].any()
        # Returns = advantages + values.
        np.testing.assert_allclose(
            buf.returns[:4],
            buf.advantages[:4] + buf.values[:4],
            atol=1e-6,
        )

    def test_terminal_state_cuts_bootstrap(self):
        buf = RolloutBuffer(
            rollout_steps=3, num_agents=1,
            obs_shape=(NUM_CHANNELS, 10, 10), num_actions=4,
        )
        for i in range(3):
            buf.add(
                states=np.zeros(
                    (1, NUM_CHANNELS, 10, 10), dtype=np.float32,
                ),
                actions=np.array([0], dtype=np.int64),
                rewards=np.array([1.0], dtype=np.float32),
                values=np.array([0.5], dtype=np.float32),
                log_probs=np.array([-0.5], dtype=np.float32),
                dones=np.array(
                    [1.0 if i == 1 else 0.0], dtype=np.float32,
                ),
                action_masks=np.ones((1, 4), dtype=bool),
            )
        last_values = np.array([0.5], dtype=np.float32)
        buf.compute_returns(last_values, gamma=0.99, gae_lambda=0.95)
        # Should have valid advantages.
        assert not np.isnan(buf.advantages[:3]).any()


class TestRolloutBufferBatches:
    def test_generate_batches(self):
        buf = RolloutBuffer(
            rollout_steps=8, num_agents=2,
            obs_shape=(NUM_CHANNELS, 10, 10), num_actions=4,
        )
        for _ in range(8):
            buf.add(
                states=np.random.randn(
                    2, NUM_CHANNELS, 10, 10,
                ).astype(np.float32),
                actions=np.random.randint(
                    0, 4, size=2,
                ).astype(np.int64),
                rewards=np.random.randn(2).astype(np.float32),
                values=np.random.randn(2).astype(np.float32),
                log_probs=np.random.randn(2).astype(np.float32),
                dones=np.zeros(2, dtype=np.float32),
                action_masks=np.ones((2, 4), dtype=bool),
            )
        last_v = np.zeros(2, dtype=np.float32)
        buf.compute_returns(last_v, gamma=0.99, gae_lambda=0.95)

        batches = buf.generate_batches(num_minibatches=4)
        assert len(batches) == 4
        # Each batch: 16 total / 4 = 4.
        for b in batches:
            assert b["states"].shape[0] == 4
            assert b["actions"].shape[0] == 4
            assert b["log_probs"].shape[0] == 4
            assert b["advantages"].shape[0] == 4
            assert b["returns"].shape[0] == 4
            assert b["action_masks"].shape == (4, 4)

    def test_reset_clears_buffer(self):
        buf = RolloutBuffer(
            rollout_steps=4, num_agents=2,
            obs_shape=(NUM_CHANNELS, 10, 10), num_actions=4,
        )
        for _ in range(4):
            buf.add(
                states=np.zeros(
                    (2, NUM_CHANNELS, 10, 10), dtype=np.float32,
                ),
                actions=np.zeros(2, dtype=np.int64),
                rewards=np.zeros(2, dtype=np.float32),
                values=np.zeros(2, dtype=np.float32),
                log_probs=np.zeros(2, dtype=np.float32),
                dones=np.zeros(2, dtype=np.float32),
                action_masks=np.ones((2, 4), dtype=bool),
            )
        assert buf.full
        buf.reset()
        assert len(buf) == 0
        assert not buf.full
