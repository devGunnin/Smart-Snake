"""MAPPO reinforcement learning agent for Smart Snake."""

from smart_snake.ai.agent import MAPPOAgent
from smart_snake.ai.benchmark import BenchmarkResult, benchmark_throughput
from smart_snake.ai.config import RewardConfig, TrainingConfig
from smart_snake.ai.difficulty import (
    DifficultyAgent,
    DifficultyConfig,
    DifficultyTier,
    TierConfig,
    load_tier_agent,
)
from smart_snake.ai.environment import MultiSnakeEnv, SnakeEnv
from smart_snake.ai.model_manager import CheckpointMeta, ModelManager
from smart_snake.ai.parallel import SubprocessVectorizedEnv, VectorizedEnv
from smart_snake.ai.replay_buffer import RolloutBuffer
from smart_snake.ai.train import SelfPlayTrainer

__all__ = [
    "BenchmarkResult",
    "CheckpointMeta",
    "DifficultyAgent",
    "DifficultyConfig",
    "DifficultyTier",
    "MAPPOAgent",
    "ModelManager",
    "MultiSnakeEnv",
    "RewardConfig",
    "RolloutBuffer",
    "SelfPlayTrainer",
    "SnakeEnv",
    "SubprocessVectorizedEnv",
    "TierConfig",
    "TrainingConfig",
    "VectorizedEnv",
    "benchmark_throughput",
    "load_tier_agent",
]
