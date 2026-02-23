"""Tests for experience replay buffers."""

import numpy as np
import pytest

from smart_snake.ai.replay_buffer import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    Transition,
)
from smart_snake.ai.state import NUM_CHANNELS


def _make_transition(i: int = 0) -> Transition:
    """Create a dummy transition for testing."""
    s = np.zeros((NUM_CHANNELS, 10, 10), dtype=np.float32)
    s[0, 0, i % 10] = 1.0
    ns = np.zeros((NUM_CHANNELS, 10, 10), dtype=np.float32)
    ns[0, 0, (i + 1) % 10] = 1.0
    return Transition(state=s, action=i % 4, reward=float(i), next_state=ns, done=False)


class TestReplayBuffer:
    def test_invalid_capacity(self):
        with pytest.raises(ValueError, match="capacity"):
            ReplayBuffer(0)

    def test_add_and_len(self):
        buf = ReplayBuffer(5)
        assert len(buf) == 0
        buf.add(_make_transition(0))
        assert len(buf) == 1

    def test_circular_overwrite(self):
        buf = ReplayBuffer(3)
        for i in range(5):
            buf.add(_make_transition(i))
        assert len(buf) == 3
        # Oldest entries (0, 1) should be overwritten.
        rewards = {t.reward for t in buf._buffer}
        assert 0.0 not in rewards
        assert 1.0 not in rewards

    def test_sample_size(self):
        buf = ReplayBuffer(10)
        for i in range(10):
            buf.add(_make_transition(i))
        batch = buf.sample(5)
        assert len(batch) == 5
        assert all(isinstance(t, Transition) for t in batch)

    def test_sample_too_large_raises(self):
        buf = ReplayBuffer(5)
        buf.add(_make_transition(0))
        with pytest.raises(ValueError, match="Cannot sample"):
            buf.sample(5)

    def test_sample_deterministic_with_rng(self):
        buf = ReplayBuffer(10)
        for i in range(10):
            buf.add(_make_transition(i))
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        b1 = buf.sample(3, rng=rng1)
        b2 = buf.sample(3, rng=rng2)
        assert [t.reward for t in b1] == [t.reward for t in b2]

    def test_capacity_property(self):
        buf = ReplayBuffer(42)
        assert buf.capacity == 42


class TestPrioritizedReplayBuffer:
    def test_invalid_capacity(self):
        with pytest.raises(ValueError, match="capacity"):
            PrioritizedReplayBuffer(0)

    def test_add_and_len(self):
        buf = PrioritizedReplayBuffer(5)
        assert len(buf) == 0
        buf.add(_make_transition(0))
        assert len(buf) == 1

    def test_circular_overwrite(self):
        buf = PrioritizedReplayBuffer(3)
        for i in range(5):
            buf.add(_make_transition(i))
        assert len(buf) == 3

    def test_sample_returns_weights_and_indices(self):
        buf = PrioritizedReplayBuffer(10)
        for i in range(10):
            buf.add(_make_transition(i))
        transitions, weights, indices = buf.sample(5, beta=0.4)
        assert len(transitions) == 5
        assert weights.shape == (5,)
        assert weights.dtype == np.float32
        assert indices.shape == (5,)
        # Weights should be in (0, 1].
        assert np.all(weights > 0)
        assert np.all(weights <= 1.0 + 1e-6)

    def test_sample_too_large_raises(self):
        buf = PrioritizedReplayBuffer(5)
        buf.add(_make_transition(0))
        with pytest.raises(ValueError, match="Cannot sample"):
            buf.sample(5)

    def test_update_priorities(self):
        buf = PrioritizedReplayBuffer(10, alpha=0.6)
        for i in range(10):
            buf.add(_make_transition(i))
        _, _, indices = buf.sample(3, beta=0.4)
        td_errors = np.array([0.5, 1.0, 2.0])
        buf.update_priorities(indices, td_errors)
        # Priorities should have been updated.
        for idx, td in zip(indices, td_errors, strict=True):
            expected = (abs(td) + 1e-6) ** 0.6
            assert buf._priorities[idx] == pytest.approx(expected, rel=1e-5)

    def test_capacity_property(self):
        buf = PrioritizedReplayBuffer(42)
        assert buf.capacity == 42
