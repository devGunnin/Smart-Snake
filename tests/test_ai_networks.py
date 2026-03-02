"""Tests for actor-critic network architecture."""

import torch

from smart_snake.ai.networks import ActorCriticNetwork
from smart_snake.ai.state import NUM_CHANNELS


class TestActorCriticNetwork:
    def test_output_shapes(self):
        net = ActorCriticNetwork(
            in_channels=NUM_CHANNELS, height=20, width=20,
            num_actions=4,
        )
        x = torch.randn(1, NUM_CHANNELS, 20, 20)
        logits, value = net(x)
        assert logits.shape == (1, 4)
        assert value.shape == (1,)

    def test_batch_output_shapes(self):
        net = ActorCriticNetwork(
            in_channels=NUM_CHANNELS, height=10, width=10,
            num_actions=4,
        )
        x = torch.randn(8, NUM_CHANNELS, 10, 10)
        logits, value = net(x)
        assert logits.shape == (8, 4)
        assert value.shape == (8,)

    def test_different_grid_size(self):
        net = ActorCriticNetwork(
            in_channels=NUM_CHANNELS, height=15, width=25,
            num_actions=4, conv_channels=(16, 32),
        )
        x = torch.randn(2, NUM_CHANNELS, 15, 25)
        logits, value = net(x)
        assert logits.shape == (2, 4)
        assert value.shape == (2,)

    def test_gradient_flow(self):
        net = ActorCriticNetwork(
            in_channels=NUM_CHANNELS, height=10, width=10,
            num_actions=4,
        )
        x = torch.randn(4, NUM_CHANNELS, 10, 10)
        logits, value = net(x)
        loss = logits.sum() + value.sum()
        loss.backward()
        for param in net.parameters():
            assert param.grad is not None

    def test_action_masking(self):
        net = ActorCriticNetwork(
            in_channels=NUM_CHANNELS, height=10, width=10,
            num_actions=4,
        )
        x = torch.randn(2, NUM_CHANNELS, 10, 10)
        mask = torch.tensor([
            [True, False, True, True],
            [True, True, False, False],
        ])
        logits, value = net(x, action_mask=mask)
        # Masked logits should be very negative.
        assert logits[0, 1].item() < -1e7
        assert logits[1, 2].item() < -1e7
        assert logits[1, 3].item() < -1e7
        # Value should still be scalar per sample.
        assert value.shape == (2,)

    def test_no_mask_passes(self):
        net = ActorCriticNetwork(
            in_channels=NUM_CHANNELS, height=10, width=10,
            num_actions=4,
        )
        x = torch.randn(2, NUM_CHANNELS, 10, 10)
        logits, value = net(x, action_mask=None)
        assert logits.shape == (2, 4)
        assert value.shape == (2,)
