"""Self-play training loop with TensorBoard metrics logging."""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import numpy as np

from smart_snake.ai.agent import DQNAgent
from smart_snake.ai.config import TrainingConfig
from smart_snake.ai.environment import MultiSnakeEnv
from smart_snake.ai.replay_buffer import Transition

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TENSORBOARD = True
except ImportError:  # pragma: no cover
    _HAS_TENSORBOARD = False


class SelfPlayTrainer:
    """Runs self-play training: one DQN agent controls all snakes.

    Each snake's experience is added to the shared replay buffer,
    giving the agent diverse perspectives of the same game.
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.agent = DQNAgent(self.config, device=device)
        self.env = MultiSnakeEnv(
            player_count=self.config.player_count,
            grid_width=self.config.grid_width,
            grid_height=self.config.grid_height,
            wall_mode=self.config.wall_mode,
            max_apples=self.config.max_apples,
            initial_snake_length=self.config.initial_snake_length,
            reward_config=self.config.reward,
            max_steps=self.config.max_steps_per_episode,
            state_encoding=self.config.state_encoding,
        )
        self._rng = np.random.default_rng()

        self._writer: SummaryWriter | None = None
        if _HAS_TENSORBOARD:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(log_dir))

        # Rolling metrics.
        self.episode_rewards: deque[float] = deque(maxlen=100)
        self.episode_lengths: deque[int] = deque(maxlen=100)
        self.episode_wins: deque[int] = deque(maxlen=100)
        self.losses: deque[float] = deque(maxlen=100)
        self.total_steps = 0
        self.total_episodes = 0

    def run_episode(self) -> dict:
        """Run a single self-play episode and return summary metrics."""
        cfg = self.config
        obs_list, _ = self.env.reset(seed=int(self._rng.integers(2**31)))
        states = list(obs_list)
        alive = [True] * self.env.num_agents
        episode_reward = [0.0] * self.env.num_agents
        steps = 0

        while True:
            actions: list[int] = []
            for sid in range(self.env.num_agents):
                a = (
                    self.agent.select_action(states[sid], rng=self._rng)
                    if alive[sid] else 0
                )
                actions.append(a)

            next_obs, rewards, terminated, truncated, info = self.env.step(actions)
            steps += 1

            for sid in range(self.env.num_agents):
                if not alive[sid]:
                    continue
                done = terminated[sid] or truncated[sid]
                self.agent.store(Transition(
                    state=states[sid],
                    action=actions[sid],
                    reward=rewards[sid],
                    next_state=next_obs[sid],
                    done=done,
                ))
                episode_reward[sid] += rewards[sid]
                if terminated[sid] or truncated[sid]:
                    alive[sid] = False

            states = list(next_obs)
            self.total_steps += 1

            if self.agent.can_train():
                loss = self.agent.train_step()
                self.losses.append(loss)

            if all(not a for a in alive) or info.get("game_over"):
                break
            if steps >= cfg.max_steps_per_episode:
                break

        mean_reward = float(np.mean(episode_reward))
        self.episode_rewards.append(mean_reward)
        self.episode_lengths.append(steps)
        self.episode_wins.append(1 if info.get("winner") is not None else 0)
        self.total_episodes += 1

        return {
            "episode": self.total_episodes,
            "steps": steps,
            "mean_reward": mean_reward,
            "scores": info.get("scores", []),
            "winner": info.get("winner"),
        }

    def train(self) -> None:
        """Run the full training loop."""
        cfg = self.config
        logger.info(
            "Starting self-play training: %d episodes, %d players.",
            cfg.max_episodes, cfg.player_count,
        )
        start = time.monotonic()

        for ep in range(1, cfg.max_episodes + 1):
            self.run_episode()

            if ep % cfg.log_interval == 0:
                elapsed = time.monotonic() - start
                avg_reward = (
                    float(np.mean(self.episode_rewards))
                    if self.episode_rewards else 0.0
                )
                avg_length = (
                    float(np.mean(self.episode_lengths))
                    if self.episode_lengths else 0.0
                )
                avg_loss = (
                    float(np.mean(self.losses)) if self.losses else 0.0
                )
                win_rate = (
                    float(np.mean(self.episode_wins))
                    if self.episode_wins else 0.0
                )

                logger.info(
                    "Episode %d | reward=%.3f | length=%.1f | loss=%.4f "
                    "| win_rate=%.2f | eps=%.3f | %.1fs",
                    ep, avg_reward, avg_length, avg_loss,
                    win_rate, self.agent.epsilon, elapsed,
                )

                if self._writer is not None:
                    self._writer.add_scalar("reward/mean", avg_reward, ep)
                    self._writer.add_scalar("episode/length", avg_length, ep)
                    self._writer.add_scalar("train/loss", avg_loss, ep)
                    self._writer.add_scalar("train/epsilon", self.agent.epsilon, ep)
                    self._writer.add_scalar("train/win_rate", win_rate, ep)

            if ep % cfg.save_interval == 0:
                ckpt = Path(cfg.checkpoint_dir) / f"dqn_ep{ep}.pt"
                self.agent.save(ckpt)

        # Final checkpoint.
        final = Path(cfg.checkpoint_dir) / "dqn_final.pt"
        self.agent.save(final)

        if self._writer is not None:
            self._writer.close()

        logger.info(
            "Training complete: %d episodes, %d total steps.",
            self.total_episodes, self.total_steps,
        )

    def close(self) -> None:
        """Release resources."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
