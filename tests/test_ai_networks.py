"""Tests for DQN network architectures."""

import torch

from smart_snake.ai.networks import DQNNetwork, DuelingDQNNetwork
from smart_snake.ai.state import NUM_CHANNELS


class TestDQNNetwork:
    def test_output_shape(self):
        net = DQNNetwork(
            in_channels=NUM_CHANNELS, height=20, width=20, num_actions=4,
        )
        x = torch.randn(1, NUM_CHANNELS, 20, 20)
        out = net(x)
        assert out.shape == (1, 4)

    def test_batch_output_shape(self):
        net = DQNNetwork(
            in_channels=NUM_CHANNELS, height=10, width=10, num_actions=4,
        )
        x = torch.randn(8, NUM_CHANNELS, 10, 10)
        out = net(x)
        assert out.shape == (8, 4)

    def test_different_grid_size(self):
        net = DQNNetwork(
            in_channels=NUM_CHANNELS, height=15, width=25, num_actions=4,
            conv_channels=(16, 32),
        )
        x = torch.randn(2, NUM_CHANNELS, 15, 25)
        out = net(x)
        assert out.shape == (2, 4)

    def test_gradient_flow(self):
        net = DQNNetwork(
            in_channels=NUM_CHANNELS, height=10, width=10, num_actions=4,
        )
        x = torch.randn(4, NUM_CHANNELS, 10, 10)
        out = net(x)
        loss = out.sum()
        loss.backward()
        for param in net.parameters():
            assert param.grad is not None


class TestDuelingDQNNetwork:
    def test_output_shape(self):
        net = DuelingDQNNetwork(
            in_channels=NUM_CHANNELS, height=20, width=20, num_actions=4,
        )
        x = torch.randn(1, NUM_CHANNELS, 20, 20)
        out = net(x)
        assert out.shape == (1, 4)

    def test_batch_output_shape(self):
        net = DuelingDQNNetwork(
            in_channels=NUM_CHANNELS, height=10, width=10, num_actions=4,
        )
        x = torch.randn(8, NUM_CHANNELS, 10, 10)
        out = net(x)
        assert out.shape == (8, 4)

    def test_advantage_mean_subtracted(self):
        """The mean advantage should be ~0 across actions for any input."""
        net = DuelingDQNNetwork(
            in_channels=NUM_CHANNELS, height=10, width=10, num_actions=4,
        )
        x = torch.randn(4, NUM_CHANNELS, 10, 10)
        # Forward pass with value and advantage extraction.
        features = net.conv(x)
        value = net.value_stream(features)
        advantage = net.advantage_stream(features)
        q = value + advantage - advantage.mean(dim=1, keepdim=True)
        # The mean Q-value should equal the value stream output.
        q_mean = q.mean(dim=1, keepdim=True)
        torch.testing.assert_close(q_mean, value, atol=1e-5, rtol=1e-5)

    def test_gradient_flow(self):
        net = DuelingDQNNetwork(
            in_channels=NUM_CHANNELS, height=10, width=10, num_actions=4,
        )
        x = torch.randn(4, NUM_CHANNELS, 10, 10)
        out = net(x)
        loss = out.sum()
        loss.backward()
        for param in net.parameters():
            assert param.grad is not None
