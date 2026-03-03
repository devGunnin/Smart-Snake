"""Hyperparameter configuration for MAPPO training."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

StateEncodingMode = Literal["absolute", "relative"]


@dataclass(frozen=True)
class RewardConfig:
    """Configurable reward weights for the snake environment."""

    apple: float = 3.0
    death: float = -3.0
    step_penalty: float = -0.01
    survival_bonus: float = 0.01
    kill_opponent: float = 1.0
    apple_approach: float = 0.1
    apple_retreat: float = -0.1


@dataclass(frozen=True)
class TrainingConfig:
    """Full training hyperparameter configuration.

    Supports JSON serialization for reproducibility.
    """

    # Environment
    grid_width: int = 20
    grid_height: int = 20
    player_count: int = 2
    wall_mode: str = "death"
    max_apples: int = 3
    initial_snake_length: int = 3
    state_encoding: StateEncodingMode = "relative"

    # Network
    conv_channels: tuple[int, ...] = (32, 64, 64)
    fc_hidden: int = 256

    # Optimiser
    learning_rate: float = 3e-4
    gamma: float = 0.99
    max_grad_norm: float = 10.0

    # PPO
    clip_ratio: float = 0.2
    gae_lambda: float = 0.95
    entropy_coeff: float = 0.01
    value_loss_coeff: float = 0.5
    ppo_epochs: int = 4
    num_minibatches: int = 4
    rollout_steps: int = 128

    # Self-play
    snapshot_interval: int = 50
    snapshot_pool_size: int = 10
    latest_vs_latest_prob: float = 0.8

    # Training loop
    max_episodes: int = 50_000
    max_steps_per_episode: int = 1_000
    log_interval: int = 100
    save_interval: int = 1_000

    # Parallel environments
    num_envs: int = 4

    # Rewards
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"

    def __post_init__(self) -> None:
        if self.state_encoding not in {"absolute", "relative"}:
            raise ValueError(
                "state_encoding must be either 'absolute' or 'relative', "
                f"got {self.state_encoding!r}.",
            )
        if self.num_envs < 1:
            raise ValueError(
                f"num_envs must be at least 1, got {self.num_envs}.",
            )
        if self.clip_ratio <= 0:
            raise ValueError(
                f"clip_ratio must be positive, got {self.clip_ratio}.",
            )
        if self.ppo_epochs < 1:
            raise ValueError(
                f"ppo_epochs must be at least 1, got {self.ppo_epochs}.",
            )
        if self.num_minibatches < 1:
            raise ValueError(
                "num_minibatches must be at least 1, "
                f"got {self.num_minibatches}.",
            )
        if self.rollout_steps < 1:
            raise ValueError(
                "rollout_steps must be at least 1, "
                f"got {self.rollout_steps}.",
            )
        if self.snapshot_interval < 0:
            raise ValueError(
                "snapshot_interval must be >= 0, "
                f"got {self.snapshot_interval}.",
            )
        if self.snapshot_pool_size < 1:
            raise ValueError(
                "snapshot_pool_size must be at least 1, "
                f"got {self.snapshot_pool_size}.",
            )
        if not 0.0 <= self.latest_vs_latest_prob <= 1.0:
            raise ValueError(
                "latest_vs_latest_prob must be in [0.0, 1.0], "
                f"got {self.latest_vs_latest_prob}.",
            )

    def to_dict(self) -> dict:
        """Serialize to a plain dict (tuples become lists)."""
        return asdict(self)

    def save(self, path: str | Path) -> None:
        """Write config to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Config saved to %s", p)

    @classmethod
    def load(cls, path: str | Path) -> TrainingConfig:
        """Load config from a JSON file."""
        raw = json.loads(Path(path).read_text())
        reward_data = raw.pop("reward", {})
        raw["reward"] = RewardConfig(**reward_data)
        if "conv_channels" in raw:
            raw["conv_channels"] = tuple(raw["conv_channels"])
        if "state_encoding" in raw and raw["state_encoding"] not in {
            "absolute",
            "relative",
        }:
            raise ValueError(
                "state_encoding must be either 'absolute' or 'relative', "
                f"got {raw['state_encoding']!r}.",
            )
        return cls(**raw)
