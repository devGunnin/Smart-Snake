"""Tests for the MAPPO self-play training loop."""

import numpy as np
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
    def test_crossed_intervals_returns_all_boundaries(self):
        assert SelfPlayTrainer._crossed_intervals(0, 25, 10) == [
            10, 20,
        ]
        assert SelfPlayTrainer._crossed_intervals(10, 25, 10) == [
            20,
        ]
        assert SelfPlayTrainer._crossed_intervals(20, 25, 10) == []

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

    def test_caps_minibatches_to_available_samples(self, tmp_path):
        cfg = _fast_config(
            max_episodes=1,
            max_steps_per_episode=1,
            num_envs=1,
            rollout_steps=16,
            num_minibatches=4,
            checkpoint_dir=str(tmp_path / "ckpts"),
            log_dir=str(tmp_path / "runs"),
        )
        trainer = SelfPlayTrainer(cfg, device="cpu")
        trainer.train()
        assert trainer.total_episodes == 1
        assert len(trainer.losses) > 0
        trainer.close()

    def test_episode_metrics_span_multiple_rollouts(
        self, monkeypatch,
    ):
        cfg = _fast_config(
            max_episodes=1,
            num_envs=1,
            rollout_steps=1,
            player_count=2,
        )
        trainer = SelfPlayTrainer(cfg, device="cpu")
        obs_shape = trainer._rollout_buffer.obs_shape

        class _LongEpisodeEnv:
            def __init__(self):
                self._steps = 0

            def reset(self, seed=None):
                del seed
                self._steps = 0
                obs = [
                    np.zeros(obs_shape, dtype=np.float32)
                    for _ in range(cfg.player_count)
                ]
                return obs, {}

            def get_action_masks(self):
                return [
                    np.ones(4, dtype=bool)
                    for _ in range(cfg.player_count)
                ]

            def step(self, _actions):
                self._steps += 1
                obs = [
                    np.zeros(obs_shape, dtype=np.float32)
                    for _ in range(cfg.player_count)
                ]
                rewards = [1.0, 3.0]
                done = self._steps >= 3
                terminated = [done] * cfg.player_count
                truncated = [False] * cfg.player_count
                info = {}
                if done:
                    info = {
                        "game_over": True,
                        "winner": 0,
                        "scores": [1.0, 2.0],
                    }
                return obs, rewards, terminated, truncated, info

        trainer._envs = [_LongEpisodeEnv()]
        monkeypatch.setattr(
            trainer.agent,
            "sample_opponent",
            lambda rng: None,
        )
        monkeypatch.setattr(
            trainer.agent,
            "select_action",
            lambda *_args, **_kwargs: (0, 0.0, 0.0),
        )

        states, _ = trainer._reset_envs()
        for _ in range(3):
            states, _ = trainer._collect_rollout(states)

        assert trainer.total_episodes == 1
        assert trainer.episode_lengths[-1] == 3
        assert trainer.episode_rewards[-1] == 6.0
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

    def test_processes_all_crossed_interval_boundaries(
        self, monkeypatch, tmp_path,
    ):
        cfg = _fast_config(
            max_episodes=25,
            log_interval=10,
            save_interval=10,
            snapshot_interval=10,
            checkpoint_dir=str(tmp_path / "ckpts"),
        )
        trainer = SelfPlayTrainer(cfg, device="cpu")

        obs = np.zeros(
            trainer._rollout_buffer.obs_shape, dtype=np.float32,
        )
        states = [
            [obs.copy() for _ in range(cfg.player_count)]
            for _ in range(cfg.num_envs)
        ]

        class _StubRolloutBuffer:
            def __init__(self, obs_shape, num_agents):
                self.obs_shape = obs_shape
                self.num_agents = num_agents

            def __len__(self):
                return 1

            def compute_returns(self, *_args, **_kwargs):
                return None

            def generate_batches(self, *_args, **_kwargs):
                return [None]

        trainer._rollout_buffer = _StubRolloutBuffer(  # type: ignore[assignment]
            trainer._rollout_buffer.obs_shape,
            trainer._num_envs,
        )

        episode_targets = iter((23, 25))
        log_calls: list[int] = []
        save_calls: list[tuple[int, bool]] = []
        snapshot_calls = {"count": 0}

        def _fake_collect(current_states):
            trainer.total_episodes = next(episode_targets)
            return current_states, 1

        monkeypatch.setattr(
            trainer, "_reset_envs", lambda: (states, []),
        )
        monkeypatch.setattr(trainer, "_collect_rollout", _fake_collect)
        monkeypatch.setattr(
            trainer.agent,
            "get_values",
            lambda *_args, **_kwargs: [0.0] * trainer._num_envs,
        )
        monkeypatch.setattr(
            trainer.agent,
            "update",
            lambda *_args, **_kwargs: {"total_loss": 0.0},
        )
        monkeypatch.setattr(
            trainer.agent,
            "save_snapshot",
            lambda: snapshot_calls.__setitem__(
                "count", snapshot_calls["count"] + 1,
            ),
        )
        monkeypatch.setattr(
            trainer, "_log_metrics", lambda ep, _start: log_calls.append(ep),
        )
        monkeypatch.setattr(
            trainer,
            "_save_versioned_checkpoint",
            lambda ep, final=False: save_calls.append((ep, final)),
        )

        trainer.train()

        assert snapshot_calls["count"] == 2
        assert log_calls == [10, 20, 25]
        assert save_calls == [(10, False), (20, False), (25, True)]
        trainer.close()

    def test_episode_scores_tracked(self):
        trainer = SelfPlayTrainer(
            _fast_config(max_episodes=5), device="cpu",
        )
        trainer.train()
        assert len(trainer.episode_scores) > 0
        trainer.close()
