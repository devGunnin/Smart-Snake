"""Difficulty tier system mapping checkpoints to skill levels."""

from __future__ import annotations

import enum
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from smart_snake.ai.config import RewardConfig, TrainingConfig
from smart_snake.ai.networks import DQNNetwork, DuelingDQNNetwork
from smart_snake.ai.state import NUM_CHANNELS

logger = logging.getLogger(__name__)


class DifficultyTier(enum.Enum):
    """Skill tiers for AI opponents, ordered by difficulty."""

    BEGINNER = "beginner"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    IMPOSSIBLE = "impossible"


DEFAULT_WIN_RATE_TARGETS: dict[DifficultyTier, float] = {
    DifficultyTier.BEGINNER: 0.0,
    DifficultyTier.EASY: 0.2,
    DifficultyTier.MEDIUM: 0.5,
    DifficultyTier.HARD: 0.75,
    DifficultyTier.IMPOSSIBLE: 0.9,
}

ALL_TIERS: list[DifficultyTier] = list(DifficultyTier)


@dataclass
class TierConfig:
    """Maps a :class:`DifficultyTier` to a checkpoint and target."""

    tier: str
    checkpoint_path: str
    win_rate_target: float
    random_action_prob: float = 0.0


@dataclass
class DifficultyConfig:
    """Full tier configuration, loadable from JSON."""

    tiers: list[TierConfig]

    def to_dict(self) -> dict:
        return {"tiers": [asdict(t) for t in self.tiers]}

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> DifficultyConfig:
        raw = json.loads(Path(path).read_text())
        tiers = [TierConfig(**t) for t in raw["tiers"]]
        return cls(tiers=tiers)

    def get_tier(self, tier: DifficultyTier | str) -> TierConfig:
        """Look up config for a specific tier."""
        name = tier.value if isinstance(tier, DifficultyTier) else tier
        for t in self.tiers:
            if t.tier == name:
                return t
        raise KeyError(f"Tier {name!r} not found in config.")


def _config_from_dict(d: dict) -> TrainingConfig:
    """Reconstruct a TrainingConfig from a checkpoint dict."""
    d = dict(d)
    reward_data = d.pop("reward", {})
    d["reward"] = RewardConfig(**reward_data)
    if "conv_channels" in d:
        d["conv_channels"] = tuple(d["conv_channels"])
    return TrainingConfig(**d)


class DifficultyAgent:
    """Inference-only agent loaded from a tier checkpoint.

    Optionally injects random actions at a configurable rate to
    lower effective skill for easier tiers.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        random_action_prob: float = 0.0,
        device: str | None = None,
    ) -> None:
        self._device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._random_action_prob = max(0.0, min(1.0, random_action_prob))
        self._rng = np.random.default_rng()

        data = torch.load(
            Path(checkpoint_path),
            map_location=self._device,
            weights_only=False,
        )
        cfg_dict = data.get("config", {})
        config = _config_from_dict(cfg_dict)

        net_cls = DuelingDQNNetwork if config.dueling else DQNNetwork
        self._net = net_cls(
            in_channels=NUM_CHANNELS,
            height=config.grid_height,
            width=config.grid_width,
            num_actions=4,
            conv_channels=config.conv_channels,
            fc_hidden=config.fc_hidden,
        ).to(self._device)
        self._net.load_state_dict(data["online_state_dict"])
        self._net.eval()

    def select_action(self, state: np.ndarray) -> int:
        """Select an action, with optional random perturbation."""
        if self._rng.random() < self._random_action_prob:
            return int(self._rng.integers(4))
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0).to(self._device)
            q = self._net(t)
            return int(q.argmax(dim=1).item())

    def select_actions_batch(
        self, states: list[np.ndarray],
    ) -> list[int]:
        """Select actions for a batch of states."""
        batch_size = len(states)
        random_mask = (
            self._rng.random(batch_size) < self._random_action_prob
        )
        with torch.no_grad():
            t = torch.from_numpy(np.stack(states)).to(self._device)
            q = self._net(t)
            greedy = q.argmax(dim=1).cpu().numpy()
        random_actions = self._rng.integers(4, size=batch_size)
        actions = np.where(random_mask, random_actions, greedy)
        return actions.tolist()


def load_tier_agent(
    difficulty_config: DifficultyConfig,
    tier: DifficultyTier | str,
    *,
    device: str | None = None,
) -> DifficultyAgent:
    """Load an inference agent for the given difficulty tier."""
    tier_cfg = difficulty_config.get_tier(tier)
    return DifficultyAgent(
        tier_cfg.checkpoint_path,
        random_action_prob=tier_cfg.random_action_prob,
        device=device,
    )
