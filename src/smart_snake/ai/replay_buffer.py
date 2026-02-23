"""Experience replay buffers for DQN training."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Transition:
    """A single experience tuple."""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size circular replay buffer with uniform sampling."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1.")
        self._capacity = capacity
        self._buffer: list[Transition] = []
        self._pos = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def add(self, transition: Transition) -> None:
        """Add a transition, overwriting the oldest if full."""
        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._pos] = transition
        self._pos = (self._pos + 1) % self._capacity

    def sample(
        self, batch_size: int, rng: np.random.Generator | None = None,
    ) -> list[Transition]:
        """Sample a uniform random batch."""
        if batch_size > len(self._buffer):
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {len(self._buffer)}."
            )
        gen = rng or np.random.default_rng()
        indices = gen.choice(len(self._buffer), size=batch_size, replace=False)
        return [self._buffer[i] for i in indices]


class PrioritizedReplayBuffer:
    """Proportional prioritized experience replay.

    Uses a sum-tree for O(log n) sampling and priority updates.
    Importance-sampling weights are computed with an annealed beta.
    """

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1.")
        self._capacity = capacity
        self._alpha = alpha
        self._buffer: list[Transition | None] = [None] * capacity
        self._priorities = np.zeros(capacity, dtype=np.float64)
        self._pos = 0
        self._size = 0
        self._max_priority = 1.0

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        return self._size

    def add(self, transition: Transition) -> None:
        """Add a transition with maximum priority."""
        self._buffer[self._pos] = transition
        self._priorities[self._pos] = self._max_priority ** self._alpha
        self._pos = (self._pos + 1) % self._capacity
        self._size = min(self._size + 1, self._capacity)

    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
        rng: np.random.Generator | None = None,
    ) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        """Sample a prioritized batch.

        Returns ``(transitions, weights, indices)`` where *weights* are
        importance-sampling corrections and *indices* identify the
        sampled entries for later priority updates.
        """
        if batch_size > self._size:
            raise ValueError(
                f"Cannot sample {batch_size} from buffer of size {self._size}."
            )
        gen = rng or np.random.default_rng()

        prios = self._priorities[:self._size]
        total = prios.sum()
        probs = prios / total

        indices = gen.choice(self._size, size=batch_size, replace=False, p=probs)

        # Importance-sampling weights.
        min_prob = prios.min() / total
        max_weight = (self._size * min_prob) ** (-beta)
        weights = (self._size * probs[indices]) ** (-beta)
        weights /= max_weight  # normalise to [0, 1]

        transitions = [self._buffer[i] for i in indices]
        return transitions, weights.astype(np.float32), indices

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray,
    ) -> None:
        """Update priorities using absolute TD errors."""
        clipped = np.abs(td_errors) + 1e-6
        for idx, prio in zip(indices, clipped, strict=True):
            self._priorities[idx] = prio ** self._alpha
            self._max_priority = max(self._max_priority, float(prio))
