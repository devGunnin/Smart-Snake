"""Tests for model management, checkpointing, and versioning."""

import json

import pytest
import torch

from smart_snake.ai.config import TrainingConfig
from smart_snake.ai.model_manager import CheckpointMeta, ModelManager


def _dummy_state_dict() -> dict:
    """Create a minimal checkpoint dict for testing."""
    return {
        "online_state_dict": {"layer.weight": torch.randn(4, 4)},
        "target_state_dict": {"layer.weight": torch.randn(4, 4)},
        "optimiser_state_dict": {},
        "step_count": 100,
        "config": TrainingConfig(
            grid_width=10, grid_height=10,
        ).to_dict(),
    }


class TestCheckpointMeta:
    def test_to_dict(self):
        meta = CheckpointMeta(
            version=1, step=100, episode=10, win_rate=0.5,
            mean_reward=1.0, timestamp="2025-01-01T00:00:00",
            config={}, filename="v0001.pt",
        )
        d = meta.to_dict()
        assert d["version"] == 1
        assert d["win_rate"] == 0.5
        assert d["filename"] == "v0001.pt"


class TestModelManager:
    def test_init_creates_dir(self, tmp_path):
        mgr = ModelManager(tmp_path / "ckpts")
        assert (tmp_path / "ckpts").is_dir()
        assert mgr.latest_version == 0
        assert mgr.versions == []

    def test_save_checkpoint(self, tmp_path):
        mgr = ModelManager(tmp_path / "ckpts")
        config = TrainingConfig(grid_width=10, grid_height=10)
        meta = mgr.save_checkpoint(
            _dummy_state_dict(),
            step=100, episode=10, win_rate=0.6,
            mean_reward=1.5, config=config,
        )
        assert meta.version == 1
        assert meta.win_rate == 0.6
        assert mgr.latest_version == 1
        assert len(mgr.versions) == 1
        assert (tmp_path / "ckpts" / meta.filename).exists()

    def test_sequential_versions(self, tmp_path):
        mgr = ModelManager(tmp_path / "ckpts")
        config = TrainingConfig(grid_width=10, grid_height=10)
        for i in range(3):
            mgr.save_checkpoint(
                _dummy_state_dict(),
                step=i * 100, episode=i * 10,
                win_rate=0.1 * i, mean_reward=0.5 * i,
                config=config,
            )
        assert mgr.latest_version == 3
        assert len(mgr.versions) == 3

    def test_best_model_tracking(self, tmp_path):
        mgr = ModelManager(tmp_path / "ckpts")
        config = TrainingConfig(grid_width=10, grid_height=10)
        mgr.save_checkpoint(
            _dummy_state_dict(), step=100, episode=10,
            win_rate=0.3, mean_reward=1.0, config=config,
        )
        mgr.save_checkpoint(
            _dummy_state_dict(), step=200, episode=20,
            win_rate=0.8, mean_reward=2.0, config=config,
        )
        mgr.save_checkpoint(
            _dummy_state_dict(), step=300, episode=30,
            win_rate=0.5, mean_reward=1.5, config=config,
        )
        assert mgr.best_win_rate == 0.8
        assert (tmp_path / "ckpts" / "best_model.pt").exists()

    def test_load_checkpoint_latest(self, tmp_path):
        mgr = ModelManager(tmp_path / "ckpts")
        config = TrainingConfig(grid_width=10, grid_height=10)
        mgr.save_checkpoint(
            _dummy_state_dict(), step=100, episode=10,
            win_rate=0.5, mean_reward=1.0, config=config,
        )
        loaded = mgr.load_checkpoint()
        assert "online_state_dict" in loaded

    def test_load_checkpoint_by_version(self, tmp_path):
        mgr = ModelManager(tmp_path / "ckpts")
        config = TrainingConfig(grid_width=10, grid_height=10)
        mgr.save_checkpoint(
            _dummy_state_dict(), step=100, episode=10,
            win_rate=0.3, mean_reward=1.0, config=config,
        )
        mgr.save_checkpoint(
            _dummy_state_dict(), step=200, episode=20,
            win_rate=0.5, mean_reward=1.5, config=config,
        )
        loaded = mgr.load_checkpoint(version=1)
        assert loaded["step_count"] == 100

    def test_load_checkpoint_not_found(self, tmp_path):
        mgr = ModelManager(tmp_path / "ckpts")
        with pytest.raises(FileNotFoundError):
            mgr.load_checkpoint()

    def test_load_checkpoint_version_not_found(self, tmp_path):
        mgr = ModelManager(tmp_path / "ckpts")
        config = TrainingConfig(grid_width=10, grid_height=10)
        mgr.save_checkpoint(
            _dummy_state_dict(), step=100, episode=10,
            win_rate=0.5, mean_reward=1.0, config=config,
        )
        with pytest.raises(FileNotFoundError, match="version 99"):
            mgr.load_checkpoint(version=99)

    def test_load_best(self, tmp_path):
        mgr = ModelManager(tmp_path / "ckpts")
        config = TrainingConfig(grid_width=10, grid_height=10)
        mgr.save_checkpoint(
            _dummy_state_dict(), step=100, episode=10,
            win_rate=0.5, mean_reward=1.0, config=config,
        )
        loaded = mgr.load_best()
        assert "online_state_dict" in loaded

    def test_load_best_not_found(self, tmp_path):
        mgr = ModelManager(tmp_path / "ckpts")
        with pytest.raises(FileNotFoundError, match="No best model"):
            mgr.load_best()

    def test_get_meta(self, tmp_path):
        mgr = ModelManager(tmp_path / "ckpts")
        config = TrainingConfig(grid_width=10, grid_height=10)
        mgr.save_checkpoint(
            _dummy_state_dict(), step=100, episode=10,
            win_rate=0.5, mean_reward=1.0, config=config,
        )
        meta = mgr.get_meta(1)
        assert meta.version == 1
        assert meta.step == 100

    def test_get_meta_not_found(self, tmp_path):
        mgr = ModelManager(tmp_path / "ckpts")
        with pytest.raises(KeyError, match="version 42"):
            mgr.get_meta(42)

    def test_metadata_persistence(self, tmp_path):
        ckpt_dir = tmp_path / "ckpts"
        config = TrainingConfig(grid_width=10, grid_height=10)
        mgr1 = ModelManager(ckpt_dir)
        mgr1.save_checkpoint(
            _dummy_state_dict(), step=100, episode=10,
            win_rate=0.5, mean_reward=1.0, config=config,
        )
        mgr2 = ModelManager(ckpt_dir)
        assert len(mgr2.versions) == 1
        assert mgr2.latest_version == 1
        assert mgr2.best_win_rate == 0.5

    def test_export_for_inference(self, tmp_path):
        state = _dummy_state_dict()
        src = tmp_path / "full.pt"
        torch.save(state, src)

        out = tmp_path / "inference.pt"
        result = ModelManager.export_for_inference(src, out)
        assert result == out
        assert out.exists()

        loaded = torch.load(out, weights_only=False)
        assert "online_state_dict" in loaded
        assert "config" in loaded
        assert "optimiser_state_dict" not in loaded
        assert "target_state_dict" not in loaded

    def test_metadata_json_is_valid(self, tmp_path):
        mgr = ModelManager(tmp_path / "ckpts")
        config = TrainingConfig(grid_width=10, grid_height=10)
        mgr.save_checkpoint(
            _dummy_state_dict(), step=100, episode=10,
            win_rate=0.5, mean_reward=1.0, config=config,
        )
        meta_path = tmp_path / "ckpts" / "metadata.json"
        assert meta_path.exists()
        data = json.loads(meta_path.read_text())
        assert "versions" in data
        assert "best_win_rate" in data
