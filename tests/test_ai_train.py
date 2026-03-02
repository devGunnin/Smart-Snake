"""Tests for the MAPPO self-play training loop."""

import pytest

from smart_snake.ai.config import TrainingConfig
from smart_snake.ai.train import SelfPlayTrainer


def _fast_config(**overrides) -> TrainingConfig:
    """Tiny config for fast smoke tests."""
    defaults = dict(
        grid_width=10,
        grid_height=10,
        player_count=2,
        conv_channels=(8,),
        fc_hidden=16,
        max_episodes=5,
        max_steps_per_episode=50,
        log_interval=2,
        save_interval=100,
        num_envs=1,
        rollout_steps=16,
        ppo_epochs=1,
        num_minibatches=1,
        clip_ratio=0.2,
        gae_lambda=0.95,
        entropy_coeff=0.01,
    )
    defaults.update(overrides)
    return TrainingConfig(**defaults)


class TestSelfPlayTrainer:
    def test_wires_state_encoding_mode(self):
        cfg = _fast_config(state_encoding="relative")
        trainer = SelfPlayTrainer(cfg, device="cpu")
        assert trainer._envs[0]._state_encoding == "relative"
        trainer.close()

    def test_train_completes(self):
        trainer = SelfPlayTrainer(_fast_config(), device="cpu")
        trainer.train()
        assert trainer.total_episodes == 5
        trainer.close()

    def test_losses_collected(self):
        cfg = _fast_config(max_episodes=10)
        trainer = SelfPlayTrainer(cfg, device="cpu")
        trainer.train()
        assert len(trainer.losses) > 0
        trainer.close()

    def test_three_player_training(self):
        cfg = _fast_config(player_count=3, max_episodes=3)
        trainer = SelfPlayTrainer(cfg, device="cpu")
        trainer.train()
        assert trainer.total_episodes == 3
        trainer.close()

    def test_parallel_envs(self):
        cfg = _fast_config(num_envs=2, max_episodes=10)
        trainer = SelfPlayTrainer(cfg, device="cpu")
        assert trainer._num_envs == 2
        trainer.train()
        assert trainer.total_episodes == 10
        trainer.close()

    def test_parallel_train_completes(self, tmp_path):
        cfg = _fast_config(
            num_envs=2, max_episodes=10,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )
        trainer = SelfPlayTrainer(cfg, device="cpu")
        trainer.train()
        assert trainer.total_episodes == 10
        trainer.close()

    def test_samples_snapshot_opponents_during_rollout(self, monkeypatch):
        cfg = _fast_config(
            max_episodes=1,
            latest_vs_latest_prob=0.0,
        )
        trainer = SelfPlayTrainer(cfg, device="cpu")
        trainer.agent.save_snapshot()

        called = {"count": 0}
        original = trainer.agent.select_action_with_policy

        def _counting_select_action_with_policy(*args, **kwargs):
            called["count"] += 1
            return original(*args, **kwargs)

        monkeypatch.setattr(
            trainer.agent,
            "select_action_with_policy",
            _counting_select_action_with_policy,
        )

        states, _ = trainer._reset_envs()
        trainer._collect_rollout(states)
        assert called["count"] > 0
        trainer.close()

    def test_rejects_invalid_num_envs(self):
        cfg = _fast_config()
        object.__setattr__(cfg, "num_envs", 0)
        with pytest.raises(
            ValueError, match="num_envs must be at least 1",
        ):
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
            max_episodes=10,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )
        trainer = SelfPlayTrainer(cfg, device="cpu")
        trainer.train()
        mgr = trainer.model_manager
        assert len(mgr.versions) > 0
        trainer.close()

    def test_episode_scores_tracked(self):
        trainer = SelfPlayTrainer(
            _fast_config(max_episodes=5), device="cpu",
        )
        trainer.train()
        assert len(trainer.episode_scores) > 0
        trainer.close()
