"""Model checkpointing, versioning, and export for inference."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch

from smart_snake.ai.config import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMeta:
    """Metadata for a saved model checkpoint."""

    version: int
    step: int
    episode: int
    win_rate: float
    mean_reward: float
    timestamp: str
    config: dict
    filename: str

    def to_dict(self) -> dict:
        return asdict(self)


class ModelManager:
    """Manages model checkpoints with versioning and metadata.

    Checkpoints are saved under ``checkpoint_dir/`` with sequential
    version numbers and a ``metadata.json`` index tracking all saved
    versions.
    """

    METADATA_FILE = "metadata.json"
    BEST_MODEL_FILE = "best_model.pt"

    def __init__(self, checkpoint_dir: str | Path) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._meta_path = self._dir / self.METADATA_FILE
        self._versions: list[CheckpointMeta] = []
        self._best_win_rate: float = -1.0
        self._load_existing_metadata()

    def _load_existing_metadata(self) -> None:
        if self._meta_path.exists():
            raw = json.loads(self._meta_path.read_text())
            self._versions = [
                CheckpointMeta(**v) for v in raw.get("versions", [])
            ]
            self._best_win_rate = raw.get("best_win_rate", -1.0)
            logger.info(
                "Loaded %d existing checkpoints from %s.",
                len(self._versions), self._meta_path,
            )

    def _save_metadata(self) -> None:
        data = {
            "versions": [v.to_dict() for v in self._versions],
            "best_win_rate": self._best_win_rate,
        }
        self._meta_path.write_text(json.dumps(data, indent=2))

    @property
    def versions(self) -> list[CheckpointMeta]:
        return list(self._versions)

    @property
    def best_win_rate(self) -> float:
        return self._best_win_rate

    @property
    def latest_version(self) -> int:
        """Return the latest version number, or 0 if none."""
        return self._versions[-1].version if self._versions else 0

    def save_checkpoint(
        self,
        state_dict: dict,
        *,
        step: int,
        episode: int,
        win_rate: float,
        mean_reward: float,
        config: TrainingConfig,
    ) -> CheckpointMeta:
        """Save a versioned checkpoint with metadata."""
        version = self.latest_version + 1
        filename = f"v{version:04d}_ep{episode}_step{step}.pt"
        path = self._dir / filename
        torch.save(state_dict, path)

        meta = CheckpointMeta(
            version=version,
            step=step,
            episode=episode,
            win_rate=win_rate,
            mean_reward=mean_reward,
            timestamp=datetime.now(timezone.utc).isoformat(),
            config=config.to_dict(),
            filename=filename,
        )
        self._versions.append(meta)

        if win_rate > self._best_win_rate:
            self._best_win_rate = win_rate
            best_path = self._dir / self.BEST_MODEL_FILE
            torch.save(state_dict, best_path)
            logger.info(
                "New best model: v%d (win_rate=%.3f).",
                version, win_rate,
            )

        self._save_metadata()
        logger.info(
            "Checkpoint v%d saved: step=%d, episode=%d, win_rate=%.3f.",
            version, step, episode, win_rate,
        )
        return meta

    def load_checkpoint(
        self, version: int | None = None, *, device: str = "cpu",
    ) -> dict:
        """Load a checkpoint by version (``None`` = latest)."""
        if not self._versions:
            raise FileNotFoundError("No checkpoints available.")
        if version is None:
            meta = self._versions[-1]
        else:
            matches = [
                v for v in self._versions if v.version == version
            ]
            if not matches:
                raise FileNotFoundError(
                    f"Checkpoint version {version} not found."
                )
            meta = matches[0]
        path = self._dir / meta.filename
        return torch.load(
            path, map_location=device, weights_only=False,
        )

    def load_best(self, *, device: str = "cpu") -> dict:
        """Load the best checkpoint (highest win rate)."""
        best_path = self._dir / self.BEST_MODEL_FILE
        if not best_path.exists():
            raise FileNotFoundError("No best model checkpoint found.")
        return torch.load(
            best_path, map_location=device, weights_only=False,
        )

    def get_meta(self, version: int) -> CheckpointMeta:
        """Get metadata for a specific version."""
        for v in self._versions:
            if v.version == version:
                return v
        raise KeyError(f"Checkpoint version {version} not found.")

    @staticmethod
    def export_for_inference(
        checkpoint_path: str | Path,
        output_path: str | Path,
        *,
        device: str = "cpu",
    ) -> Path:
        """Export a checkpoint stripped to inference-only weights.

        Removes optimizer state and step count, keeping only the
        online network weights and config.
        """
        data = torch.load(
            Path(checkpoint_path),
            map_location=device,
            weights_only=False,
        )
        inference_data = {
            "online_state_dict": data["online_state_dict"],
            "config": data.get("config", {}),
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(inference_data, out)
        logger.info("Exported inference model to %s.", out)
        return out
