"""Hyperparameter configuration for DQN training."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewardConfig:
    """Configurable reward weights for the snake environment."""

    apple: float = 1.0
    death: float = -1.0
    step_penalty: float = -0.01
    survival_bonus: float = 0.0
    kill_opponent: float = 0.5


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

    # Network
    dueling: bool = True
    double_dqn: bool = True
    conv_channels: tuple[int, ...] = (32, 64, 64)
    fc_hidden: int = 256

    # Optimiser
    learning_rate: float = 1e-4
    gamma: float = 0.99
    batch_size: int = 64
    max_grad_norm: float = 10.0

    # Replay buffer
    buffer_size: int = 100_000
    min_buffer_size: int = 1_000
    prioritized_replay: bool = True
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_end: float = 1.0
    priority_beta_steps: int = 100_000

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay_steps: int = 100_000

    # Target network
    target_update_freq: int = 1_000

    # Training loop
    max_episodes: int = 10_000
    max_steps_per_episode: int = 1_000
    log_interval: int = 100
    save_interval: int = 1_000

    # Rewards
    reward: RewardConfig = field(default_factory=RewardConfig)

    # Paths
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "runs"

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
        return cls(**raw)
