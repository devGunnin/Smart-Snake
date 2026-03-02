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
            num_agents=self.config.player_count * self._num_envs,
            obs_shape=obs_shape,
            num_actions=4,
        )

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
        for env in self._envs:
            obs, info = env.reset(
                seed=int(self._rng.integers(2**31)),
            )
            all_obs.append(list(obs))
            all_info.append(info)
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
        num_agents = cfg.player_count
        completed_episodes = 0
        env_steps = [0] * self._num_envs
        env_rewards: list[list[float]] = [
            [0.0] * num_agents for _ in range(self._num_envs)
        ]

        self._rollout_buffer.reset()

        for _step in range(cfg.rollout_steps):
            # Flatten observations across envs and agents.
            flat_states: list[np.ndarray] = []
            flat_masks: list[np.ndarray] = []
            for ei in range(self._num_envs):
                masks = self._envs[ei].get_action_masks()
                for sid in range(num_agents):
                    flat_states.append(states[ei][sid])
                    flat_masks.append(masks[sid])

            actions, log_probs, values = (
                self.agent.select_actions_batch(
                    flat_states, action_masks=flat_masks,
                )
            )

            # Reshape back to (env, agent).
            all_actions: list[list[int]] = []
            all_log_probs: list[list[float]] = []
            all_values: list[list[float]] = []
            all_masks_np: list[list[np.ndarray]] = []
            idx = 0
            for _ei in range(self._num_envs):
                env_acts: list[int] = []
                env_lps: list[float] = []
                env_vals: list[float] = []
                env_msks: list[np.ndarray] = []
                for _sid in range(num_agents):
                    env_acts.append(actions[idx])
                    env_lps.append(log_probs[idx])
                    env_vals.append(values[idx])
                    env_msks.append(flat_masks[idx])
                    idx += 1
                all_actions.append(env_acts)
                all_log_probs.append(env_lps)
                all_values.append(env_vals)
                all_masks_np.append(env_msks)

            # Step each environment.
            step_states = np.zeros(
                (self._num_envs * num_agents, *self._rollout_buffer.obs_shape),
                dtype=np.float32,
            )
            step_actions = np.zeros(
                self._num_envs * num_agents, dtype=np.int64,
            )
            step_rewards = np.zeros(
                self._num_envs * num_agents, dtype=np.float32,
            )
            step_values = np.zeros(
                self._num_envs * num_agents, dtype=np.float32,
            )
            step_log_probs = np.zeros(
                self._num_envs * num_agents, dtype=np.float32,
            )
            step_dones = np.zeros(
                self._num_envs * num_agents, dtype=np.float32,
            )
            step_masks = np.ones(
                (self._num_envs * num_agents, 4), dtype=bool,
            )

            for ei in range(self._num_envs):
                next_obs, rewards, terminated, truncated, info = (
                    self._envs[ei].step(all_actions[ei])
                )
                env_steps[ei] += 1
                self.total_steps += 1

                for sid in range(num_agents):
                    flat_idx = ei * num_agents + sid
                    step_states[flat_idx] = states[ei][sid]
                    step_actions[flat_idx] = all_actions[ei][sid]
                    step_rewards[flat_idx] = rewards[sid]
                    step_values[flat_idx] = all_values[ei][sid]
                    step_log_probs[flat_idx] = all_log_probs[ei][sid]
                    done = terminated[sid] or truncated[sid]
                    step_dones[flat_idx] = float(done)
                    step_masks[flat_idx] = all_masks_np[ei][sid]
                    env_rewards[ei][sid] += rewards[sid]

                states[ei] = list(next_obs)

                # Handle episode completion.
                if info.get("game_over") or all(
                    terminated[s] or truncated[s]
                    for s in range(num_agents)
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
                    self.total_episodes += 1
                    completed_episodes += 1

                    # Reset this env.
                    obs, _ = self._envs[ei].reset(
                        seed=int(self._rng.integers(2**31)),
                    )
                    states[ei] = list(obs)
                    env_steps[ei] = 0
                    env_rewards[ei] = [0.0] * num_agents

            self._rollout_buffer.add(
                states=step_states.reshape(
                    self._num_envs * num_agents,
                    *self._rollout_buffer.obs_shape,
                ),
                actions=step_actions,
                rewards=step_rewards,
                values=step_values,
                log_probs=step_log_probs,
                dones=step_dones,
                action_masks=step_masks,
            )

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

        while self.total_episodes < cfg.max_episodes:
            # Collect rollout.
            states, _completed = self._collect_rollout(states)

            # Bootstrap value for last state.
            flat_last: list[np.ndarray] = []
            for ei in range(self._num_envs):
                for sid in range(cfg.player_count):
                    flat_last.append(states[ei][sid])
            last_values = np.array(
                self.agent.get_values(flat_last), dtype=np.float32,
            )

            self._rollout_buffer.compute_returns(
                last_values, cfg.gamma, cfg.gae_lambda,
            )

            # PPO update: multiple epochs over mini-batches.
            for _epoch in range(cfg.ppo_epochs):
                batches = self._rollout_buffer.generate_batches(
                    cfg.num_minibatches, rng=self._rng,
                )
                metrics = self.agent.update(batches)
                self.losses.append(metrics["total_loss"])

            # Snapshot for self-play pool.
            if (
                cfg.snapshot_interval > 0
                and self.total_episodes > 0
                and self.total_episodes % cfg.snapshot_interval == 0
            ):
                self.agent.save_snapshot()

            # Logging.
            if self._crossed_interval(
                prev_log_ep, self.total_episodes, cfg.log_interval,
            ) or self.total_episodes >= cfg.max_episodes:
                self._log_metrics(self.total_episodes, start)
                prev_log_ep = self.total_episodes

            # Checkpoint saving.
            if self._crossed_interval(
                prev_save_ep, self.total_episodes, cfg.save_interval,
            ):
                self._save_versioned_checkpoint(
                    self.total_episodes,
                )
                prev_save_ep = self.total_episodes

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
