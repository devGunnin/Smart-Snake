"""Neural network architectures for MAPPO actor-critic agents."""

from __future__ import annotations

import torch
import torch.nn as nn


class ActorCriticNetwork(nn.Module):
    """CNN-based actor-critic network for MAPPO.

    Shared convolutional encoder feeds two separate heads:
    - **Actor**: outputs action logits (supports action masking).
    - **Critic**: outputs a scalar state value (centralized critic
      that receives the full global observation).
    """

    def __init__(
        self,
        in_channels: int,
        height: int,
        width: int,
        num_actions: int,
        conv_channels: tuple[int, ...] = (32, 64, 64),
        fc_hidden: int = 256,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        c_in = in_channels
        for c_out in conv_channels:
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1))
            layers.append(nn.ReLU())
            c_in = c_out
        self.conv = nn.Sequential(*layers)
        flat_size = conv_channels[-1] * height * width

        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(action_logits, state_value)``.

        Parameters
        ----------
        x:
            Observation tensor ``(batch, C, H, W)``.
        action_mask:
            Optional boolean tensor ``(batch, num_actions)`` where
            ``True`` marks *valid* actions.  Invalid actions receive
            ``-1e8`` logit bias before softmax.
        """
        features = self.conv(x)
        logits = self.actor(features)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e8)
        value = self.critic(features).squeeze(-1)
        return logits, value
