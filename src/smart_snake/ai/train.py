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
from smart_snake.ai.model_manager import ModelManager
from smart_snake.ai.parallel import VectorizedEnv
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

    When ``config.num_envs > 1``, uses a :class:`VectorizedEnv` to
    run multiple games in parallel with batched inference.
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.agent = DQNAgent(self.config, device=device)
        self._num_envs = max(1, self.config.num_envs)

        env_kwargs = dict(
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

        if self._num_envs > 1:
            self._vec_env: VectorizedEnv | None = VectorizedEnv(
                self._num_envs, **env_kwargs,
            )
            self.env = self._vec_env.envs[0]
        else:
            self._vec_env = None
            self.env = MultiSnakeEnv(**env_kwargs)

        self._rng = np.random.default_rng()

        self._writer: SummaryWriter | None = None
        if _HAS_TENSORBOARD:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(log_dir))

        self._model_manager = ModelManager(self.config.checkpoint_dir)

        # Rolling metrics.
        self.episode_rewards: deque[float] = deque(maxlen=100)
        self.episode_lengths: deque[int] = deque(maxlen=100)
        self.episode_wins: deque[int] = deque(maxlen=100)
        self.losses: deque[float] = deque(maxlen=100)
        self.total_steps = 0
        self.total_episodes = 0

    @property
    def model_manager(self) -> ModelManager:
        return self._model_manager

    def run_episode(self) -> dict:
        """Run a single self-play episode and return summary metrics."""
        cfg = self.config
        obs_list, _ = self.env.reset(
            seed=int(self._rng.integers(2**31)),
        )
        states = list(obs_list)
        alive = [True] * self.env.num_agents
        episode_reward = [0.0] * self.env.num_agents
        steps = 0

        while True:
            actions: list[int] = []
            for sid in range(self.env.num_agents):
                a = (
                    self.agent.select_action(
                        states[sid], rng=self._rng,
                    )
                    if alive[sid] else 0
                )
                actions.append(a)

            next_obs, rewards, terminated, truncated, info = (
                self.env.step(actions)
            )
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
        self.episode_wins.append(
            1 if info.get("winner") is not None else 0,
        )
        self.total_episodes += 1

        return {
            "episode": self.total_episodes,
            "steps": steps,
            "mean_reward": mean_reward,
            "scores": info.get("scores", []),
            "winner": info.get("winner"),
        }

    def run_parallel_episodes(
        self, *, num_envs: int | None = None,
    ) -> list[dict]:
        """Run one episode per parallel environment concurrently.

        Uses batched inference for action selection across all
        environments and agents.
        """
        if self._vec_env is None:
            return [self.run_episode()]

        cfg = self.config
        n = num_envs if num_envs is not None else self._num_envs
        if not 1 <= n <= self._num_envs:
            raise ValueError(
                f"num_envs must be in [1, {self._num_envs}], got {n}.",
            )
        num_agents = self._vec_env.num_agents

        seeds = [
            int(self._rng.integers(2**31)) for _ in range(n)
        ]
        if n == self._num_envs:
            all_obs, _ = self._vec_env.reset_all(seeds=seeds)
        else:
            all_obs = []
            for ei in range(n):
                obs, _ = self._vec_env.envs[ei].reset(seed=seeds[ei])
                all_obs.append(obs)

        states = [list(obs) for obs in all_obs]
        alive = [[True] * num_agents for _ in range(n)]
        episode_reward = [[0.0] * num_agents for _ in range(n)]
        step_counts = [0] * n
        env_done = [False] * n
        env_info: list[dict] = [{} for _ in range(n)]

        while not all(env_done):
            # Collect alive-agent states for batched inference.
            flat_states: list[np.ndarray] = []
            flat_keys: list[tuple[int, int]] = []
            for ei in range(n):
                if env_done[ei]:
                    continue
                for sid in range(num_agents):
                    if alive[ei][sid]:
                        flat_states.append(states[ei][sid])
                        flat_keys.append((ei, sid))

            if flat_states:
                flat_actions = self.agent.select_actions_batch(
                    flat_states, rng=self._rng,
                )
            else:
                flat_actions = []

            action_map: dict[tuple[int, int], int] = dict(
                zip(flat_keys, flat_actions, strict=True),
            )
            actions_per_env: list[list[int]] = []
            for ei in range(n):
                env_actions: list[int] = []
                for sid in range(num_agents):
                    env_actions.append(
                        action_map.get((ei, sid), 0),
                    )
                actions_per_env.append(env_actions)

            for ei in range(n):
                if env_done[ei]:
                    continue
                next_obs, rewards, terminated, truncated, info = (
                    self._vec_env.envs[ei].step(actions_per_env[ei])
                )
                step_counts[ei] += 1

                for sid in range(num_agents):
                    if not alive[ei][sid]:
                        continue
                    done = terminated[sid] or truncated[sid]
                    self.agent.store(Transition(
                        state=states[ei][sid],
                        action=actions_per_env[ei][sid],
                        reward=rewards[sid],
                        next_state=next_obs[sid],
                        done=done,
                    ))
                    episode_reward[ei][sid] += rewards[sid]
                    if done:
                        alive[ei][sid] = False

                states[ei] = list(next_obs)
                self.total_steps += 1
                env_info[ei] = info

                if self.agent.can_train():
                    loss = self.agent.train_step()
                    self.losses.append(loss)

                if (
                    all(not a for a in alive[ei])
                    or info.get("game_over")
                    or step_counts[ei] >= cfg.max_steps_per_episode
                ):
                    env_done[ei] = True

        results: list[dict] = []
        for ei in range(n):
            mean_r = float(np.mean(episode_reward[ei]))
            self.episode_rewards.append(mean_r)
            self.episode_lengths.append(step_counts[ei])
            self.episode_wins.append(
                1 if env_info[ei].get("winner") is not None else 0,
            )
            self.total_episodes += 1
            results.append({
                "episode": self.total_episodes,
                "steps": step_counts[ei],
                "mean_reward": mean_r,
                "scores": env_info[ei].get("scores", []),
                "winner": env_info[ei].get("winner"),
            })
        return results

    def train(self) -> None:
        """Run the full training loop."""
        cfg = self.config
        logger.info(
            "Starting self-play training: %d episodes, %d players, "
            "%d parallel env(s).",
            cfg.max_episodes, cfg.player_count, self._num_envs,
        )
        start = time.monotonic()

        ep = 0
        while ep < cfg.max_episodes:
            if self._num_envs > 1:
                remaining = cfg.max_episodes - ep
                results = self.run_parallel_episodes(
                    num_envs=min(self._num_envs, remaining),
                )
                ep += len(results)
            else:
                self.run_episode()
                ep += 1

            if ep % cfg.log_interval == 0 or ep >= cfg.max_episodes:
                self._log_metrics(ep, start)

            if ep % cfg.save_interval == 0:
                self._save_versioned_checkpoint(ep)

        # Final checkpoint.
        self._save_versioned_checkpoint(ep, final=True)

        if self._writer is not None:
            self._writer.close()

        logger.info(
            "Training complete: %d episodes, %d total steps.",
            self.total_episodes, self.total_steps,
        )

    def _log_metrics(self, ep: int, start: float) -> None:
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
            self._writer.add_scalar(
                "train/epsilon", self.agent.epsilon, ep,
            )
            self._writer.add_scalar("train/win_rate", win_rate, ep)

    def _save_versioned_checkpoint(
        self, ep: int, *, final: bool = False,
    ) -> None:
        state_dict = {
            "online_state_dict": (
                self.agent.online_net.state_dict()
            ),
            "target_state_dict": (
                self.agent.target_net.state_dict()
            ),
            "optimiser_state_dict": (
                self.agent.optimiser.state_dict()
            ),
            "step_count": self.agent._step_count,
            "config": self.config.to_dict(),
        }

        win_rate = (
            float(np.mean(self.episode_wins))
            if self.episode_wins else 0.0
        )
        mean_reward = (
            float(np.mean(self.episode_rewards))
            if self.episode_rewards else 0.0
        )

        self._model_manager.save_checkpoint(
            state_dict,
            step=self.agent._step_count,
            episode=ep,
            win_rate=win_rate,
            mean_reward=mean_reward,
            config=self.config,
        )

        # Legacy-format checkpoint for backward compatibility.
        ckpt_dir = Path(self.config.checkpoint_dir)
        if final:
            self.agent.save(ckpt_dir / "dqn_final.pt")
        else:
            self.agent.save(ckpt_dir / f"dqn_ep{ep}.pt")

    def close(self) -> None:
        """Release resources."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
