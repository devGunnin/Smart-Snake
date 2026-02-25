"""Tests for the self-play training loop."""

import pytest

from smart_snake.ai.config import TrainingConfig
from smart_snake.ai.train import SelfPlayTrainer


def _fast_config(**overrides) -> TrainingConfig:
    """Tiny config for fast smoke tests."""
    defaults = dict(
        grid_width=10,
        grid_height=10,
        player_count=2,
        batch_size=4,
        buffer_size=100,
        min_buffer_size=8,
        target_update_freq=5,
        conv_channels=(8,),
        fc_hidden=16,
        epsilon_decay_steps=50,
        max_episodes=5,
        max_steps_per_episode=50,
        log_interval=2,
        save_interval=100,
        prioritized_replay=False,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


class TestSelfPlayTrainer:
    def test_wires_state_encoding_mode(self):
        cfg = _fast_config(state_encoding="relative")
        trainer = SelfPlayTrainer(cfg, device="cpu")
        assert trainer.env._state_encoding == "relative"
        trainer.close()

    def test_run_episode_returns_metrics(self):
        trainer = SelfPlayTrainer(_fast_config(), device="cpu")
        metrics = trainer.run_episode()
        assert "episode" in metrics
        assert "steps" in metrics
        assert "mean_reward" in metrics
        assert metrics["episode"] == 1
        assert metrics["steps"] > 0
        trainer.close()

    def test_multiple_episodes(self):
        trainer = SelfPlayTrainer(_fast_config(), device="cpu")
        for _ in range(3):
            trainer.run_episode()
        assert trainer.total_episodes == 3
        assert trainer.total_steps > 0
        assert len(trainer.episode_rewards) == 3
        trainer.close()

    def test_train_completes(self):
        trainer = SelfPlayTrainer(_fast_config(), device="cpu")
        trainer.train()
        assert trainer.total_episodes == 5
        trainer.close()

    def test_losses_collected_after_warmup(self):
        cfg = _fast_config()
        trainer = SelfPlayTrainer(cfg, device="cpu")
        # Run enough episodes to fill the buffer past min_buffer_size.
        for _ in range(10):
            trainer.run_episode()
        # After sufficient experience, losses should have been recorded.
        assert len(trainer.losses) > 0
        trainer.close()

    def test_three_player_training(self):
        cfg = TrainingConfig(
            grid_width=10,
            grid_height=10,
            player_count=3,
            batch_size=4,
            buffer_size=100,
            min_buffer_size=8,
            target_update_freq=5,
            conv_channels=(8,),
            fc_hidden=16,
            max_episodes=3,
            max_steps_per_episode=30,
            prioritized_replay=False,
        )
        trainer = SelfPlayTrainer(cfg, device="cpu")
        metrics = trainer.run_episode()
        assert metrics["steps"] > 0
        trainer.close()

    def test_parallel_envs(self):
        cfg = _fast_config(num_envs=2, max_episodes=4)
        trainer = SelfPlayTrainer(cfg, device="cpu")
        assert trainer._num_envs == 2
        assert trainer._vec_env is not None
        results = trainer.run_parallel_episodes()
        assert len(results) == 2
        assert trainer.total_episodes == 2
        trainer.close()

    def test_parallel_train_completes(self, tmp_path):
        cfg = _fast_config(
            num_envs=2, max_episodes=4,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )
        trainer = SelfPlayTrainer(cfg, device="cpu")
        trainer.train()
        assert trainer.total_episodes == 4
        trainer.close()

    def test_parallel_train_respects_max_episodes(self, tmp_path):
        cfg = _fast_config(
            num_envs=2, max_episodes=5,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )
        trainer = SelfPlayTrainer(cfg, device="cpu")
        trainer.train()
        assert trainer.total_episodes == 5
        trainer.close()

    def test_rejects_invalid_num_envs(self):
        cfg = _fast_config()
        object.__setattr__(cfg, "num_envs", 0)
        with pytest.raises(ValueError, match="num_envs must be at least 1"):
            SelfPlayTrainer(cfg, device="cpu")

    def test_model_manager_exposed(self, tmp_path):
        cfg = _fast_config(
            checkpoint_dir=str(tmp_path / "ckpts"),
        )
        trainer = SelfPlayTrainer(cfg, device="cpu")
        assert trainer.model_manager is not None
        trainer.close()

    def test_versioned_checkpoints_created(self, tmp_path):
        cfg = _fast_config(
            save_interval=2,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )
        trainer = SelfPlayTrainer(cfg, device="cpu")
        trainer.train()
        mgr = trainer.model_manager
        assert len(mgr.versions) > 0
        trainer.close()
