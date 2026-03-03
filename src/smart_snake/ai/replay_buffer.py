"""On-policy rollout buffer for MAPPO training."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class RolloutBuffer:
    """Fixed-length trajectory buffer for on-policy PPO updates.

    Stores ``rollout_steps`` transitions per agent across ``num_agents``
    agents.  After a full rollout, call :meth:`compute_returns` to
    compute GAE advantages, then iterate mini-batches via
    :meth:`generate_batches`.
    """

    def __init__(
        self,
        rollout_steps: int,
        num_agents: int,
        obs_shape: tuple[int, ...],
        num_actions: int,
    ) -> None:
        if rollout_steps < 1:
            raise ValueError(
                f"rollout_steps must be at least 1, got {rollout_steps}.",
            )
        if num_agents < 1:
            raise ValueError(
                f"num_agents must be at least 1, got {num_agents}.",
            )
        self.rollout_steps = rollout_steps
        self.num_agents = num_agents
        self.obs_shape = obs_shape
        self.num_actions = num_actions

        self.states = np.zeros(
            (rollout_steps, num_agents, *obs_shape), dtype=np.float32,
        )
        self.actions = np.zeros(
            (rollout_steps, num_agents), dtype=np.int64,
        )
        self.rewards = np.zeros(
            (rollout_steps, num_agents), dtype=np.float32,
        )
        self.values = np.zeros(
            (rollout_steps, num_agents), dtype=np.float32,
        )
        self.log_probs = np.zeros(
            (rollout_steps, num_agents), dtype=np.float32,
        )
        self.dones = np.zeros(
            (rollout_steps, num_agents), dtype=np.float32,
        )
        self.action_masks = np.ones(
            (rollout_steps, num_agents, num_actions), dtype=bool,
        )

        self.advantages = np.zeros(
            (rollout_steps, num_agents), dtype=np.float32,
        )
        self.returns = np.zeros(
            (rollout_steps, num_agents), dtype=np.float32,
        )

        self._step = 0

    def __len__(self) -> int:
        return self._step

    @property
    def full(self) -> bool:
        """Whether the buffer has collected ``rollout_steps`` transitions."""
        return self._step >= self.rollout_steps

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        values: np.ndarray,
        log_probs: np.ndarray,
        dones: np.ndarray,
        action_masks: np.ndarray,
    ) -> None:
        """Store one timestep of data for all agents.

        Each array has shape ``(num_agents, ...)``.
        """
        if self._step >= self.rollout_steps:
            raise RuntimeError("RolloutBuffer is full; call reset().")
        self.states[self._step] = states
        self.actions[self._step] = actions
        self.rewards[self._step] = rewards
        self.values[self._step] = values
        self.log_probs[self._step] = log_probs
        self.dones[self._step] = dones
        self.action_masks[self._step] = action_masks
        self._step += 1

    def compute_returns(
        self,
        last_values: np.ndarray,
        gamma: float,
        gae_lambda: float,
    ) -> None:
        """Compute GAE advantages and discounted returns.

        Parameters
        ----------
        last_values:
            Value estimates for the state *after* the last stored step,
            shape ``(num_agents,)``.
        gamma:
            Discount factor.
        gae_lambda:
            GAE lambda for bias-variance trade-off.
        """
        gae = np.zeros(self.num_agents, dtype=np.float32)
        for t in reversed(range(self._step)):
            next_values = (
                last_values if t == self._step - 1
                else self.values[t + 1]
            )
            next_non_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * next_values * next_non_terminal
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            self.advantages[t] = gae
        self.returns[:self._step] = (
            self.advantages[:self._step] + self.values[:self._step]
        )

    def generate_batches(
        self,
        num_minibatches: int,
        rng: np.random.Generator | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Yield shuffled mini-batches from the stored rollout.

        Returns a list of dicts, each containing flat arrays for
        ``states``, ``actions``, ``log_probs``, ``advantages``,
        ``returns``, and ``action_masks``.
        """
        if num_minibatches < 1:
            raise ValueError(
                "num_minibatches must be at least 1, "
                f"got {num_minibatches}.",
            )
        gen = rng or np.random.default_rng()
        total = self._step * self.num_agents
        if total < 1:
            raise ValueError("Cannot generate batches from an empty buffer.")
        if num_minibatches > total:
            raise ValueError(
                "num_minibatches must be <= number of collected samples: "
                f"{num_minibatches} > {total}.",
            )
        indices = gen.permutation(total)

        flat_states = self.states[:self._step].reshape(
            total, *self.obs_shape,
        )
        flat_actions = self.actions[:self._step].reshape(total)
        flat_log_probs = self.log_probs[:self._step].reshape(total)
        flat_advantages = self.advantages[:self._step].reshape(total)
        flat_returns = self.returns[:self._step].reshape(total)
        flat_masks = self.action_masks[:self._step].reshape(
            total, self.num_actions,
        )

        batches: list[dict[str, np.ndarray]] = []
        for idx in np.array_split(indices, num_minibatches):
            batches.append({
                "states": flat_states[idx],
                "actions": flat_actions[idx],
                "log_probs": flat_log_probs[idx],
                "advantages": flat_advantages[idx],
                "returns": flat_returns[idx],
                "action_masks": flat_masks[idx],
            })
        return batches

    def reset(self) -> None:
        """Clear the buffer for the next rollout."""
        self._step = 0
