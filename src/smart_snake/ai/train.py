"""Self-play MAPPO training loop with TensorBoard metrics logging."""

from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path

import numpy as np

from smart_snake.ai.agent import MAPPOAgent
from smart_snake.ai.config import TrainingConfig
from smart_snake.ai.environment import MultiSnakeEnv
from smart_snake.ai.model_manager import ModelManager
from smart_snake.ai.networks import ActorCriticNetwork
from smart_snake.ai.replay_buffer import RolloutBuffer

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter

    _HAS_TENSORBOARD = True
except ImportError:  # pragma: no cover
    _HAS_TENSORBOARD = False


class SelfPlayTrainer:
    """Runs MAPPO self-play training with parameter-shared agents.

    Collects on-policy rollouts from vectorized environments, computes
    GAE advantages, and performs PPO mini-batch updates.  Periodically
    saves opponent snapshots for self-play diversity.
    """

    def __init__(
        self,
        config: TrainingConfig | None = None,
        device: str | None = None,
    ) -> None:
        self.config = config or TrainingConfig()
        self.agent = MAPPOAgent(self.config, device=device)
        if self.config.num_envs < 1:
            raise ValueError(
                f"num_envs must be at least 1, "
                f"got {self.config.num_envs}.",
            )
        self._num_envs = self.config.num_envs

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

        self._envs = [
            MultiSnakeEnv(**env_kwargs)
            for _ in range(self._num_envs)
        ]

        obs_shape = self._envs[0].observation_space.shape
        self._rollout_buffer = RolloutBuffer(
            rollout_steps=self.config.rollout_steps,
            num_agents=self._num_envs,
            obs_shape=obs_shape,
            num_actions=4,
        )
        self._learner_ids = [0] * self._num_envs
        self._opponent_policies: list[ActorCriticNetwork | None] = [
            None
        ] * self._num_envs

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
        self.episode_scores: deque[float] = deque(maxlen=100)
        self.losses: deque[float] = deque(maxlen=100)
        self.total_steps = 0
        self.total_episodes = 0

    @property
    def model_manager(self) -> ModelManager:
        return self._model_manager

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def _reset_envs(self) -> tuple[list[list[np.ndarray]], list[dict]]:
        """Reset all environments and return stacked observations."""
        all_obs: list[list[np.ndarray]] = []
        all_info: list[dict] = []
        for ei, env in enumerate(self._envs):
            obs, info = env.reset(
                seed=int(self._rng.integers(2**31)),
            )
            all_obs.append(list(obs))
            all_info.append(info)
            self._learner_ids[ei] = int(
                self._rng.integers(self.config.player_count),
            )
            self._opponent_policies[ei] = self.agent.sample_opponent(
                rng=self._rng,
            )
        return all_obs, all_info

    def _collect_rollout(
        self,
        states: list[list[np.ndarray]],
    ) -> tuple[list[list[np.ndarray]], int]:
        """Collect one rollout of ``rollout_steps`` transitions.

        Returns the final observation states and the number of
        completed episodes during this rollout.
        """
        cfg = self.config
        num_players = cfg.player_count
        completed_episodes = 0
        env_steps = [0] * self._num_envs
        env_rewards: list[list[float]] = [
            [0.0] * num_players for _ in range(self._num_envs)
        ]

        self._rollout_buffer.reset()

        for _step in range(cfg.rollout_steps):
            # Step each environment and collect learner-only transitions.
            step_states = np.zeros(
                (self._num_envs, *self._rollout_buffer.obs_shape),
                dtype=np.float32,
            )
            step_actions = np.zeros(
                self._num_envs, dtype=np.int64,
            )
            step_rewards = np.zeros(
                self._num_envs, dtype=np.float32,
            )
            step_values = np.zeros(
                self._num_envs, dtype=np.float32,
            )
            step_log_probs = np.zeros(
                self._num_envs, dtype=np.float32,
            )
            step_dones = np.zeros(
                self._num_envs, dtype=np.float32,
            )
            step_masks = np.ones(
                (self._num_envs, 4), dtype=bool,
            )

            for ei in range(self._num_envs):
                masks = self._envs[ei].get_action_masks()
                learner_sid = self._learner_ids[ei]
                learner_action, learner_log_prob, learner_value = (
                    self.agent.select_action(
                        states[ei][learner_sid],
                        action_mask=masks[learner_sid],
                        rng=self._rng,
                    )
                )
                env_actions: list[int] = [0] * num_players
                env_actions[learner_sid] = learner_action

                opponent_policy = self._opponent_policies[ei]
                for sid in range(num_players):
                    if sid == learner_sid:
                        continue
                    if opponent_policy is None:
                        action, _, _ = self.agent.select_action(
                            states[ei][sid],
                            action_mask=masks[sid],
                            rng=self._rng,
                        )
                    else:
                        action = self.agent.select_action_with_policy(
                            opponent_policy,
                            states[ei][sid],
                            action_mask=masks[sid],
                        )
                    env_actions[sid] = action

                next_obs, rewards, terminated, truncated, info = (
                    self._envs[ei].step(env_actions)
                )
                env_steps[ei] += 1
                self.total_steps += 1

                for sid in range(num_players):
                    env_rewards[ei][sid] += rewards[sid]

                step_states[ei] = states[ei][learner_sid]
                step_actions[ei] = learner_action
                step_rewards[ei] = rewards[learner_sid]
                step_values[ei] = learner_value
                step_log_probs[ei] = learner_log_prob
                learner_done = terminated[learner_sid] or truncated[learner_sid]
                step_dones[ei] = float(learner_done)
                step_masks[ei] = masks[learner_sid]

                states[ei] = list(next_obs)

                # Handle episode completion.
                if info.get("game_over") or all(
                    terminated[s] or truncated[s]
                    for s in range(num_players)
                ):
                    mean_r = float(np.mean(env_rewards[ei]))
                    self.episode_rewards.append(mean_r)
                    self.episode_lengths.append(env_steps[ei])
                    self.episode_wins.append(
                        1 if info.get("winner") is not None else 0,
                    )
                    scores = info.get("scores", [])
                    mean_sc = (
                        float(np.mean(scores)) if scores else 0.0
                    )
                    self.episode_scores.append(mean_sc)
                    if self.total_episodes < cfg.max_episodes:
                        self.total_episodes += 1
                        completed_episodes += 1

                    if self.total_episodes < cfg.max_episodes:
                        # Reset this env for continued collection.
                        obs, _ = self._envs[ei].reset(
                            seed=int(self._rng.integers(2**31)),
                        )
                        states[ei] = list(obs)
                        env_steps[ei] = 0
                        env_rewards[ei] = [0.0] * num_players
                        self._learner_ids[ei] = int(
                            self._rng.integers(num_players),
                        )
                        self._opponent_policies[ei] = (
                            self.agent.sample_opponent(
                                rng=self._rng,
                            )
                        )

            self._rollout_buffer.add(
                states=step_states,
                actions=step_actions,
                rewards=step_rewards,
                values=step_values,
                log_probs=step_log_probs,
                dones=step_dones,
                action_masks=step_masks,
            )
            if self.total_episodes >= cfg.max_episodes:
                break

        return states, completed_episodes

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full MAPPO training loop."""
        cfg = self.config
        logger.info(
            "Starting MAPPO self-play training: %d episodes, "
            "%d players, %d parallel env(s), %d rollout steps.",
            cfg.max_episodes, cfg.player_count,
            self._num_envs, cfg.rollout_steps,
        )
        start = time.monotonic()

        states, _ = self._reset_envs()

        prev_log_ep = 0
        prev_save_ep = 0
        prev_snapshot_ep = 0

        while self.total_episodes < cfg.max_episodes:
            # Collect rollout.
            states, _completed = self._collect_rollout(states)
            if len(self._rollout_buffer) == 0:
                break

            # Bootstrap value for last state.
            flat_last: list[np.ndarray] = []
            for ei in range(self._num_envs):
                learner_sid = self._learner_ids[ei]
                flat_last.append(states[ei][learner_sid])
            last_values = np.array(
                self.agent.get_values(flat_last), dtype=np.float32,
            )

            self._rollout_buffer.compute_returns(
                last_values, cfg.gamma, cfg.gae_lambda,
            )

            # PPO update: multiple epochs over mini-batches.
            num_samples = (
                len(self._rollout_buffer)
                * self._rollout_buffer.num_agents
            )
            effective_minibatches = min(
                cfg.num_minibatches, num_samples,
            )
            if effective_minibatches < cfg.num_minibatches:
                logger.warning(
                    "Capping num_minibatches to collected samples: "
                    "requested=%d, effective=%d, samples=%d.",
                    cfg.num_minibatches,
                    effective_minibatches,
                    num_samples,
                )
            for _epoch in range(cfg.ppo_epochs):
                batches = self._rollout_buffer.generate_batches(
                    effective_minibatches, rng=self._rng,
                )
                metrics = self.agent.update(batches)
                self.losses.append(metrics["total_loss"])

            # Snapshot for self-play pool.
            if cfg.snapshot_interval > 0:
                for snapshot_ep in self._crossed_intervals(
                    prev_snapshot_ep,
                    self.total_episodes,
                    cfg.snapshot_interval,
                ):
                    self.agent.save_snapshot()
                    prev_snapshot_ep = snapshot_ep

            # Logging.
            for log_ep in self._crossed_intervals(
                prev_log_ep,
                self.total_episodes,
                cfg.log_interval,
            ):
                self._log_metrics(log_ep, start)
                prev_log_ep = log_ep
            if (
                self.total_episodes >= cfg.max_episodes
                and prev_log_ep < self.total_episodes
            ):
                self._log_metrics(self.total_episodes, start)
                prev_log_ep = self.total_episodes

            # Checkpoint saving.
            for save_ep in self._crossed_intervals(
                prev_save_ep,
                self.total_episodes,
                cfg.save_interval,
            ):
                self._save_versioned_checkpoint(save_ep)
                prev_save_ep = save_ep

        # Final checkpoint.
        self._save_versioned_checkpoint(
            self.total_episodes, final=True,
        )

        if self._writer is not None:
            self._writer.close()

        logger.info(
            "Training complete: %d episodes, %d total steps.",
            self.total_episodes, self.total_steps,
        )

    @staticmethod
    def _crossed_interval(
        prev_episode: int, current_episode: int, interval: int,
    ) -> bool:
        """Return True if any interval boundary was crossed."""
        if interval < 1:
            raise ValueError(
                f"interval must be at least 1, got {interval}.",
            )
        return (
            prev_episode // interval
            < current_episode // interval
        )

    @classmethod
    def _crossed_intervals(
        cls, prev_episode: int, current_episode: int, interval: int,
    ) -> list[int]:
        """Return all crossed interval boundaries in ascending order."""
        if not cls._crossed_interval(
            prev_episode, current_episode, interval,
        ):
            return []
        first = ((prev_episode // interval) + 1) * interval
        return list(range(first, current_episode + 1, interval))

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
        avg_score = (
            float(np.mean(self.episode_scores))
            if self.episode_scores else 0.0
        )
        eps_per_sec = ep / max(elapsed, 1e-6)

        logger.info(
            "Episode %d | reward=%.3f | score=%.2f | length=%.1f "
            "| loss=%.4f | win_rate=%.3f | %.1fs (%.1f ep/s)",
            ep, avg_reward, avg_score, avg_length, avg_loss,
            win_rate, elapsed, eps_per_sec,
        )

        if self._writer is not None:
            self._writer.add_scalar("reward/mean", avg_reward, ep)
            self._writer.add_scalar("score/mean", avg_score, ep)
            self._writer.add_scalar("episode/length", avg_length, ep)
            self._writer.add_scalar("train/loss", avg_loss, ep)
            self._writer.add_scalar("train/win_rate", win_rate, ep)
            self._writer.add_scalar(
                "throughput/episodes_per_sec", eps_per_sec, ep,
            )

    def _save_versioned_checkpoint(
        self, ep: int, *, final: bool = False,
    ) -> None:
        state_dict = {
            "actor_state_dict": (
                self.agent.network.state_dict()
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

        # Legacy-format checkpoint.
        ckpt_dir = Path(self.config.checkpoint_dir)
        if final:
            self.agent.save(ckpt_dir / "mappo_final.pt")
        else:
            self.agent.save(ckpt_dir / f"mappo_ep{ep}.pt")

    def close(self) -> None:
        """Release resources."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None
