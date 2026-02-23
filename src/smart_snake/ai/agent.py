"""DQN agent with Double-DQN support, target network, and checkpointing."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as func

from smart_snake.ai.config import TrainingConfig
from smart_snake.ai.networks import DQNNetwork, DuelingDQNNetwork
from smart_snake.ai.replay_buffer import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    Transition,
)
from smart_snake.ai.state import NUM_CHANNELS

logger = logging.getLogger(__name__)


class DQNAgent:
    """DQN agent managing network, target network, and replay buffer.

    Supports Double DQN, Dueling DQN, and prioritized experience replay
    depending on the provided :class:`TrainingConfig`.
    """

    def __init__(
        self,
        config: TrainingConfig,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        net_cls = DuelingDQNNetwork if config.dueling else DQNNetwork
        net_kwargs = dict(
            in_channels=NUM_CHANNELS,
            height=config.grid_height,
            width=config.grid_width,
            num_actions=4,
            conv_channels=config.conv_channels,
            fc_hidden=config.fc_hidden,
        )
        self.online_net = net_cls(**net_kwargs).to(self.device)
        self.target_net = net_cls(**net_kwargs).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimiser = torch.optim.Adam(
            self.online_net.parameters(), lr=config.learning_rate,
        )

        if config.prioritized_replay:
            self.buffer: ReplayBuffer | PrioritizedReplayBuffer = (
                PrioritizedReplayBuffer(
                    config.buffer_size, alpha=config.priority_alpha,
                )
            )
        else:
            self.buffer = ReplayBuffer(config.buffer_size)

        self._step_count = 0

    @property
    def epsilon(self) -> float:
        """Current epsilon for epsilon-greedy exploration."""
        cfg = self.config
        frac = min(self._step_count / max(cfg.epsilon_decay_steps, 1), 1.0)
        return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)

    @property
    def beta(self) -> float:
        """Current beta for importance-sampling (prioritized replay)."""
        cfg = self.config
        frac = min(self._step_count / max(cfg.priority_beta_steps, 1), 1.0)
        return cfg.priority_beta_start + frac * (
            cfg.priority_beta_end - cfg.priority_beta_start
        )

    def select_action(
        self,
        state: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> int:
        """Epsilon-greedy action selection."""
        gen = rng or np.random.default_rng()
        if gen.random() < self.epsilon:
            return int(gen.integers(4))
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            q = self.online_net(t)
            return int(q.argmax(dim=1).item())

    def store(self, transition: Transition) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.add(transition)

    def can_train(self) -> bool:
        """Check whether the buffer has enough samples."""
        return len(self.buffer) >= self.config.min_buffer_size

    def train_step(self) -> float:
        """Run a single training step; returns the loss value."""
        cfg = self.config
        self._step_count += 1

        if isinstance(self.buffer, PrioritizedReplayBuffer):
            transitions, weights_np, indices = self.buffer.sample(
                cfg.batch_size, beta=self.beta,
            )
            weights = torch.from_numpy(weights_np).to(self.device)
        else:
            transitions = self.buffer.sample(cfg.batch_size)
            weights = None
            indices = None

        states = torch.from_numpy(
            np.stack([t.state for t in transitions]),
        ).to(self.device)
        actions = torch.tensor(
            [t.action for t in transitions], dtype=torch.long, device=self.device,
        )
        rewards = torch.tensor(
            [t.reward for t in transitions], dtype=torch.float32, device=self.device,
        )
        next_states = torch.from_numpy(
            np.stack([t.next_state for t in transitions]),
        ).to(self.device)
        dones = torch.tensor(
            [t.done for t in transitions], dtype=torch.float32, device=self.device,
        )

        # Current Q-values.
        q_values = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values (Double DQN or standard).
        with torch.no_grad():
            if cfg.double_dqn:
                next_actions = self.online_net(next_states).argmax(dim=1)
                next_q = self.target_net(next_states).gather(
                    1, next_actions.unsqueeze(1),
                ).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1).values
            target = rewards + cfg.gamma * next_q * (1.0 - dones)

        td_errors = q_values - target

        if weights is not None:
            loss = (weights * func.smooth_l1_loss(
                q_values, target, reduction="none",
            )).mean()
        else:
            loss = func.smooth_l1_loss(q_values, target)

        self.optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.online_net.parameters(), cfg.max_grad_norm,
        )
        self.optimiser.step()

        # Update priorities.
        if isinstance(self.buffer, PrioritizedReplayBuffer) and indices is not None:
            self.buffer.update_priorities(
                indices, td_errors.detach().cpu().numpy(),
            )

        # Periodic target network update.
        if self._step_count % cfg.target_update_freq == 0:
            self.sync_target()

        return float(loss.item())

    def sync_target(self) -> None:
        """Copy online network weights to the target network."""
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "online_state_dict": self.online_net.state_dict(),
                "target_state_dict": self.target_net.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "step_count": self._step_count,
                "config": self.config.to_dict(),
            },
            p,
        )
        logger.info("Checkpoint saved to %s (step %d).", p, self._step_count)

    def load(self, path: str | Path) -> None:
        """Load model checkpoint."""
        data = torch.load(Path(path), map_location=self.device, weights_only=False)
        self.online_net.load_state_dict(data["online_state_dict"])
        self.target_net.load_state_dict(data["target_state_dict"])
        self.optimiser.load_state_dict(data["optimiser_state_dict"])
        self._step_count = data["step_count"]
        logger.info("Checkpoint loaded from %s (step %d).", path, self._step_count)
