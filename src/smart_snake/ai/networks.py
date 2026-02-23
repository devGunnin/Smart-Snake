"""Neural network architectures for DQN agents."""

from __future__ import annotations

import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """Standard CNN-based DQN.

    Three convolutional layers followed by a fully-connected head that
    outputs Q-values for each action.
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
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values ``(batch, num_actions)``."""
        return self.fc(self.conv(x))


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN: separates value and advantage streams.

    ``Q(s, a) = V(s) + A(s, a) - mean(A(s, ·))``
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

        self.value_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values ``(batch, num_actions)``."""
        features = self.conv(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)
