"""CLI training launcher for Smart Snake AI."""

from __future__ import annotations

import argparse
import logging
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smart-snake-train",
        description=(
            "Smart Snake AI training, benchmarking, and export tools."
        ),
    )
    sub = parser.add_subparsers(dest="command", help="Available commands.")

    # --- train ---
    train_p = sub.add_parser("train", help="Run DQN self-play training.")
    train_p.add_argument(
        "--config", type=str, default=None,
        help="Path to a JSON config file (overrides other flags).",
    )
    train_p.add_argument("--episodes", type=int, default=None)
    train_p.add_argument("--grid-width", type=int, default=None)
    train_p.add_argument("--grid-height", type=int, default=None)
    train_p.add_argument("--players", type=int, default=None)
    train_p.add_argument("--num-envs", type=int, default=None)
    train_p.add_argument("--learning-rate", type=float, default=None)
    train_p.add_argument("--batch-size", type=int, default=None)
    train_p.add_argument("--gamma", type=float, default=None)
    train_p.add_argument("--epsilon-start", type=float, default=None)
    train_p.add_argument("--epsilon-end", type=float, default=None)
    train_p.add_argument(
        "--epsilon-decay-steps", type=int, default=None,
    )
    train_p.add_argument("--buffer-size", type=int, default=None)
    train_p.add_argument("--save-interval", type=int, default=None)
    train_p.add_argument("--log-interval", type=int, default=None)
    train_p.add_argument("--checkpoint-dir", type=str, default=None)
    train_p.add_argument("--log-dir", type=str, default=None)
    train_p.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint to resume from.",
    )
    train_p.add_argument("--device", type=str, default=None)
    train_p.add_argument(
        "--state-encoding", type=str, default=None,
        choices=["absolute", "relative"],
    )

    # --- benchmark ---
    bench_p = sub.add_parser(
        "benchmark", help="Measure simulation throughput.",
    )
    bench_p.add_argument("--num-envs", type=int, default=1)
    bench_p.add_argument("--num-games", type=int, default=100)
    bench_p.add_argument("--players", type=int, default=2)
    bench_p.add_argument("--grid-width", type=int, default=20)
    bench_p.add_argument("--grid-height", type=int, default=20)
    bench_p.add_argument("--max-steps", type=int, default=200)

    # --- export ---
    export_p = sub.add_parser(
        "export", help="Export a checkpoint for inference.",
    )
    export_p.add_argument(
        "checkpoint", help="Path to the source checkpoint.",
    )
    export_p.add_argument("output", help="Path for the exported model.")

    return parser


def _run_train(args: argparse.Namespace) -> int:
    from smart_snake.ai.config import RewardConfig, TrainingConfig
    from smart_snake.ai.train import SelfPlayTrainer

    config = (
        TrainingConfig.load(args.config)
        if args.config else TrainingConfig()
    )

    overrides: dict = {}
    flag_map = {
        "episodes": "max_episodes",
        "grid_width": "grid_width",
        "grid_height": "grid_height",
        "players": "player_count",
        "num_envs": "num_envs",
        "learning_rate": "learning_rate",
        "batch_size": "batch_size",
        "gamma": "gamma",
        "epsilon_start": "epsilon_start",
        "epsilon_end": "epsilon_end",
        "epsilon_decay_steps": "epsilon_decay_steps",
        "buffer_size": "buffer_size",
        "save_interval": "save_interval",
        "log_interval": "log_interval",
        "checkpoint_dir": "checkpoint_dir",
        "log_dir": "log_dir",
        "state_encoding": "state_encoding",
    }
    for cli_name, cfg_name in flag_map.items():
        val = getattr(args, cli_name, None)
        if val is not None:
            overrides[cfg_name] = val

    if overrides:
        d = config.to_dict()
        d.update(overrides)
        reward_data = d.pop("reward", {})
        d["reward"] = RewardConfig(**reward_data)
        if "conv_channels" in d:
            d["conv_channels"] = tuple(d["conv_channels"])
        config = TrainingConfig(**d)

    trainer = SelfPlayTrainer(config, device=args.device)

    if args.resume:
        trainer.agent.load(args.resume)
        logger.info("Resumed from checkpoint: %s", args.resume)

    try:
        trainer.train()
    finally:
        trainer.close()

    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    from smart_snake.ai.benchmark import benchmark_throughput

    result = benchmark_throughput(
        num_envs=args.num_envs,
        num_games=args.num_games,
        player_count=args.players,
        grid_width=args.grid_width,
        grid_height=args.grid_height,
        max_steps=args.max_steps,
    )
    print(result.summary())  # noqa: T201
    return 0


def _run_export(args: argparse.Namespace) -> int:
    from smart_snake.ai.model_manager import ModelManager

    out = ModelManager.export_for_inference(
        args.checkpoint, args.output,
    )
    print(f"Exported inference model to {out}")  # noqa: T201
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``smart-snake-train`` CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1

    handlers = {
        "train": _run_train,
        "benchmark": _run_benchmark,
        "export": _run_export,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
