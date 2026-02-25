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
            "--save-interval", "100",
            "--log-interval", "1",
            "--checkpoint-dir", ckpt_dir,
            "--log-dir", log_dir,
            "--device", "cpu",
        ])
        assert result == 0


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
        from smart_snake.ai.agent import DQNAgent
        from smart_snake.ai.config import TrainingConfig

        cfg = TrainingConfig(
            grid_width=10, grid_height=10,
            conv_channels=(8,), fc_hidden=16,
        )
        agent = DQNAgent(cfg, device="cpu")
        src = tmp_path / "model.pt"
        agent.save(src)

        out = tmp_path / "exported.pt"
        result = main(["export", str(src), str(out)])
        assert result == 0
        assert out.exists()
