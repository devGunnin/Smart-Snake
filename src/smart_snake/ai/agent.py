"""MAPPO agent with shared actor-critic, GAE, and self-play snapshot pool."""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from smart_snake.ai.config import TrainingConfig
from smart_snake.ai.networks import ActorCriticNetwork
from smart_snake.ai.state import NUM_CHANNELS

logger = logging.getLogger(__name__)


class MAPPOAgent:
    """Multi-Agent PPO agent with parameter sharing.

    All agents share a single :class:`ActorCriticNetwork`.  The critic
    acts as a centralized value function receiving each agent's own
    observation (parameter-shared MAPPO).

    Maintains an opponent snapshot pool for self-play diversity.
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

        self.network = ActorCriticNetwork(
            in_channels=NUM_CHANNELS,
            height=config.grid_height,
            width=config.grid_width,
            num_actions=4,
            conv_channels=config.conv_channels,
            fc_hidden=config.fc_hidden,
        ).to(self.device)

        self.optimiser = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate,
        )

        self._step_count = 0

        # Opponent snapshot pool for self-play.
        self._snapshot_pool: list[dict] = []
        self._snapshot_networks: list[ActorCriticNetwork] = []

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, float, float]:
        """Select an action using the current policy.

        Returns ``(action, log_prob, value)``.
        """
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).to(
                self.device, dtype=torch.float32,
            )
            mask_t = None
            if action_mask is not None:
                mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(
                    self.device, dtype=torch.bool,
                )
            logits, value = self.network(t, action_mask=mask_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return (
            int(action.item()),
            float(log_prob.item()),
            float(value.item()),
        )

    def select_action_with_policy(
        self,
        policy: ActorCriticNetwork,
        state: np.ndarray,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Sample one action from an explicit policy network."""
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).to(
                self.device, dtype=torch.float32,
            )
            mask_t = None
            if action_mask is not None:
                mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(
                    self.device, dtype=torch.bool,
                )
            logits, _ = policy(t, action_mask=mask_t)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
        return int(action.item())

    def select_actions_batch(
        self,
        states: list[np.ndarray],
        action_masks: list[np.ndarray] | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[list[int], list[float], list[float]]:
        """Batched action selection.

        Returns ``(actions, log_probs, values)``.
        """
        with torch.no_grad():
            t = torch.from_numpy(np.stack(states)).to(
                self.device, dtype=torch.float32,
            )
            mask_t = None
            if action_masks is not None:
                mask_t = torch.from_numpy(np.stack(action_masks)).to(
                    self.device, dtype=torch.bool,
                )
            logits, values = self.network(t, action_mask=mask_t)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        return (
            actions.cpu().tolist(),
            log_probs.cpu().tolist(),
            values.cpu().tolist(),
        )

    def get_values(self, states: list[np.ndarray]) -> list[float]:
        """Return value estimates for a batch of states."""
        with torch.no_grad():
            t = torch.from_numpy(np.stack(states)).to(
                self.device, dtype=torch.float32,
            )
            _, values = self.network(t)
        return values.cpu().tolist()

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def update(
        self,
        batches: list[dict[str, np.ndarray]],
    ) -> dict[str, float]:
        """Run one PPO update epoch over the given mini-batches.

        Returns a dict of mean metrics: ``policy_loss``,
        ``value_loss``, ``entropy``, ``total_loss``, ``clip_fraction``.
        """
        cfg = self.config
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        num_batches = 0

        for batch in batches:
            states_t = torch.from_numpy(batch["states"]).to(
                self.device, dtype=torch.float32,
            )
            actions_t = torch.from_numpy(batch["actions"]).to(
                self.device, dtype=torch.long,
            )
            old_log_probs_t = torch.from_numpy(
                batch["log_probs"],
            ).to(self.device, dtype=torch.float32)
            advantages_t = torch.from_numpy(
                batch["advantages"],
            ).to(self.device, dtype=torch.float32)
            returns_t = torch.from_numpy(batch["returns"]).to(
                self.device, dtype=torch.float32,
            )
            mask_t = torch.from_numpy(batch["action_masks"]).to(
                self.device, dtype=torch.bool,
            )

            # Normalize advantages.
            if advantages_t.numel() > 1:
                advantages_t = (
                    (advantages_t - advantages_t.mean())
                    / (advantages_t.std() + 1e-8)
                )

            logits, values = self.network(states_t, action_mask=mask_t)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - old_log_probs_t)
            surr1 = ratio * advantages_t
            surr2 = (
                torch.clamp(
                    ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio,
                )
                * advantages_t
            )
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = nn.functional.mse_loss(values, returns_t)

            loss = (
                policy_loss
                + cfg.value_loss_coeff * value_loss
                - cfg.entropy_coeff * entropy
            )

            self.optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.network.parameters(), cfg.max_grad_norm,
            )
            self.optimiser.step()
            self._step_count += 1

            clip_frac = float(
                ((ratio - 1.0).abs() > cfg.clip_ratio).float().mean(),
            )
            total_policy_loss += float(policy_loss.item())
            total_value_loss += float(value_loss.item())
            total_entropy += float(entropy.item())
            total_clip_frac += clip_frac
            num_batches += 1

        n = max(num_batches, 1)
        return {
            "policy_loss": total_policy_loss / n,
            "value_loss": total_value_loss / n,
            "entropy": total_entropy / n,
            "total_loss": (total_policy_loss + total_value_loss) / n,
            "clip_fraction": total_clip_frac / n,
        }

    # ------------------------------------------------------------------
    # Snapshot pool (self-play)
    # ------------------------------------------------------------------

    def save_snapshot(self) -> None:
        """Save a copy of the current policy to the snapshot pool."""
        snapshot = copy.deepcopy(self.network.state_dict())
        pool_size = self.config.snapshot_pool_size
        if len(self._snapshot_pool) >= pool_size:
            self._snapshot_pool.pop(0)
            self._snapshot_networks.pop(0)
        self._snapshot_pool.append(snapshot)
        net = ActorCriticNetwork(
            in_channels=NUM_CHANNELS,
            height=self.config.grid_height,
            width=self.config.grid_width,
            num_actions=4,
            conv_channels=self.config.conv_channels,
            fc_hidden=self.config.fc_hidden,
        ).to(self.device)
        net.load_state_dict(snapshot)
        net.eval()
        self._snapshot_networks.append(net)

    def sample_opponent(
        self, rng: np.random.Generator | None = None,
    ) -> ActorCriticNetwork | None:
        """Sample an opponent from the snapshot pool.

        Returns ``None`` when the pool is empty (use latest policy).
        With probability ``latest_vs_latest_prob``, returns ``None``
        to indicate use of the current policy as opponent.
        """
        gen = rng or np.random.default_rng()
        if (
            not self._snapshot_networks
            or gen.random() < self.config.latest_vs_latest_prob
        ):
            return None
        idx = int(gen.integers(len(self._snapshot_networks)))
        return self._snapshot_networks[idx]

    @property
    def snapshot_pool_size(self) -> int:
        """Number of snapshots in the pool."""
        return len(self._snapshot_pool)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor_state_dict": self.network.state_dict(),
                "optimiser_state_dict": self.optimiser.state_dict(),
                "step_count": self._step_count,
                "config": self.config.to_dict(),
            },
            p,
        )
        logger.info(
            "Checkpoint saved to %s (step %d).", p, self._step_count,
        )

    def load(self, path: str | Path) -> None:
        """Load model checkpoint."""
        data = torch.load(
            Path(path), map_location=self.device, weights_only=False,
        )
        self.network.load_state_dict(data["actor_state_dict"])
        self.optimiser.load_state_dict(data["optimiser_state_dict"])
        self._step_count = data["step_count"]
        logger.info(
            "Checkpoint loaded from %s (step %d).",
            path, self._step_count,
        )
