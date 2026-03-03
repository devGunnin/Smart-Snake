"""Tests for the CLI training launcher."""

import pytest

from smart_snake.ai.cli import _build_parser, main


class TestCLIParser:
    def test_no_command_returns_1(self):
        assert main([]) == 1

    def test_train_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["train"])
        assert args.command == "train"
        assert args.config is None
        assert args.episodes is None
        assert args.device is None

    def test_train_with_flags(self):
        parser = _build_parser()
        args = parser.parse_args([
            "train",
            "--episodes", "500",
            "--grid-width", "15",
            "--players", "3",
            "--num-envs", "4",
            "--device", "cpu",
        ])
        assert args.episodes == 500
        assert args.grid_width == 15
        assert args.players == 3
        assert args.num_envs == 4
        assert args.device == "cpu"

    def test_train_ppo_flags(self):
        parser = _build_parser()
        args = parser.parse_args([
            "train",
            "--clip-ratio", "0.3",
            "--gae-lambda", "0.9",
            "--entropy-coeff", "0.02",
            "--ppo-epochs", "3",
            "--num-minibatches", "8",
            "--rollout-steps", "64",
        ])
        assert args.clip_ratio == 0.3
        assert args.gae_lambda == 0.9
        assert args.entropy_coeff == 0.02
        assert args.ppo_epochs == 3
        assert args.num_minibatches == 8
        assert args.rollout_steps == 64

    def test_train_self_play_flags(self):
        parser = _build_parser()
        args = parser.parse_args([
            "train",
            "--snapshot-interval", "100",
            "--snapshot-pool-size", "20",
        ])
        assert args.snapshot_interval == 100
        assert args.snapshot_pool_size == 20

    def test_train_reward_flags(self):
        parser = _build_parser()
        args = parser.parse_args([
            "train",
            "--reward-apple", "5.0",
            "--reward-death", "-5.0",
            "--reward-step-penalty", "-0.05",
            "--reward-apple-approach", "0.2",
            "--reward-apple-retreat", "-0.3",
        ])
        assert args.reward_apple == 5.0
        assert args.reward_death == -5.0
        assert args.reward_step_penalty == -0.05
        assert args.reward_apple_approach == 0.2
        assert args.reward_apple_retreat == -0.3

    def test_train_extra_flags(self):
        parser = _build_parser()
        args = parser.parse_args([
            "train",
            "--max-steps-per-episode", "300",
        ])
        assert args.max_steps_per_episode == 300

    def test_benchmark_defaults(self):
        parser = _build_parser()
        args = parser.parse_args(["benchmark"])
        assert args.command == "benchmark"
        assert args.num_envs == 1
        assert args.num_games == 100

    def test_export_args(self):
        parser = _build_parser()
        args = parser.parse_args(["export", "model.pt", "output.pt"])
        assert args.command == "export"
        assert args.checkpoint == "model.pt"
        assert args.output == "output.pt"


class TestCLITrain:
    def test_train_short_run(self, tmp_path):
        """Smoke test: short training session via CLI."""
        ckpt_dir = str(tmp_path / "ckpts")
        log_dir = str(tmp_path / "logs")
        result = main([
            "train",
            "--episodes", "2",
            "--grid-width", "10",
            "--grid-height", "10",
            "--num-envs", "1",
            "--save-interval", "100",
            "--log-interval", "1",
            "--checkpoint-dir", ckpt_dir,
            "--log-dir", log_dir,
            "--device", "cpu",
            "--rollout-steps", "8",
            "--ppo-epochs", "1",
            "--num-minibatches", "1",
        ])
        assert result == 0

    def test_train_with_reward_overrides(self, tmp_path):
        ckpt_dir = str(tmp_path / "ckpts")
        log_dir = str(tmp_path / "logs")
        result = main([
            "train",
            "--episodes", "2",
            "--grid-width", "10",
            "--grid-height", "10",
            "--num-envs", "1",
            "--save-interval", "100",
            "--log-interval", "1",
            "--checkpoint-dir", ckpt_dir,
            "--log-dir", log_dir,
            "--device", "cpu",
            "--reward-apple", "5.0",
            "--reward-death", "-5.0",
            "--rollout-steps", "8",
            "--ppo-epochs", "1",
            "--num-minibatches", "1",
        ])
        assert result == 0

    def test_train_num_envs_must_be_positive(self):
        with pytest.raises(SystemExit, match="2"):
            main(["train", "--num-envs", "0"])

    def test_train_ppo_epochs_must_be_positive(self):
        with pytest.raises(SystemExit, match="2"):
            main(["train", "--ppo-epochs", "0"])


class TestCLIBenchmark:
    def test_benchmark_runs(self, capsys):
        result = main([
            "benchmark",
            "--num-games", "5",
            "--grid-width", "10",
            "--grid-height", "10",
            "--max-steps", "20",
        ])
        assert result == 0
        captured = capsys.readouterr()
        assert "Benchmark:" in captured.out
        assert "games/s" in captured.out

    def test_benchmark_num_envs_must_be_positive(self):
        with pytest.raises(SystemExit, match="2"):
            main(["benchmark", "--num-envs", "0"])


class TestCLIExport:
    def test_export(self, tmp_path):
        from smart_snake.ai.agent import MAPPOAgent
        from smart_snake.ai.config import TrainingConfig

        cfg = TrainingConfig(
            grid_width=10, grid_height=10,
            conv_channels=(8,), fc_hidden=16,
        )
        agent = MAPPOAgent(cfg, device="cpu")
        src = tmp_path / "model.pt"
        agent.save(src)

        out = tmp_path / "exported.pt"
        result = main(["export", str(src), str(out)])
        assert result == 0
        assert out.exists()
