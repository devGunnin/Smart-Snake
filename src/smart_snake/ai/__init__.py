"""DQN reinforcement learning agent for Smart Snake."""

from smart_snake.ai.agent import DQNAgent
from smart_snake.ai.config import RewardConfig, TrainingConfig
from smart_snake.ai.environment import MultiSnakeEnv, SnakeEnv
from smart_snake.ai.replay_buffer import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    Transition,
)
from smart_snake.ai.train import SelfPlayTrainer

__all__ = [
    "DQNAgent",
    "MultiSnakeEnv",
    "PrioritizedReplayBuffer",
    "ReplayBuffer",
    "RewardConfig",
    "SelfPlayTrainer",
    "SnakeEnv",
    "TrainingConfig",
    "Transition",
]
