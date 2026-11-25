# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch

from neural_rx.model import (
    ChannelReadout,
    LlrReadout,
    SeparableConv2d,
    StateInitializer,
    StateUpdater,
    UserStateAggregator,
)


def test_separable_convolution_preserves_spatial_shape() -> None:
    layer = SeparableConv2d(6, 9, kernel_size=3)

    output = layer(torch.randn(2, 6, 5, 7))

    assert output.shape == (2, 9, 5, 7)


def test_state_initializer_combines_received_pilots_and_channel_estimate() -> None:
    layer = StateInitializer(
        num_rx_antennas=2,
        state_dim=8,
        hidden_units=(12,),
        convolution="sepconv",
        initial_channel_estimate=True,
    )
    received = torch.randn(3, 5, 4, 4)
    pilot_features = torch.randn(3, 2, 2, 5, 4)
    initial_channel = torch.randn(3, 2, 5, 4, 4)

    state = layer(received, pilot_features, initial_channel)

    assert state.shape == (6, 8, 5, 4)


def test_user_state_aggregation_ignores_inactive_users() -> None:
    layer = UserStateAggregator(state_dim=1, hidden_units=())
    with torch.no_grad():
        layer.output.weight.fill_(1.0)
        layer.output.bias.zero_()
    state = torch.tensor([[[[2.0]]], [[[6.0]]], [[[100.0]]]])
    active_users = torch.tensor([[1.0, 1.0, 0.0]])

    aggregated = layer(state, active_users)

    torch.testing.assert_close(aggregated[0], torch.tensor([[[6.0]]]))
    torch.testing.assert_close(aggregated[1], torch.tensor([[[2.0]]]))


def test_zero_state_update_is_residual_identity() -> None:
    layer = StateUpdater(state_dim=4, hidden_units=(6,), convolution="conv")
    for parameter in layer.parameters():
        torch.nn.init.zeros_(parameter)
    state = torch.randn(6, 4, 5, 4)
    aggregated = torch.randn_like(state)
    pilot_features = torch.randn(6, 2, 5, 4)

    updated = layer(state, aggregated, pilot_features)

    torch.testing.assert_close(updated, state)


def test_readouts_preserve_grid_dimensions() -> None:
    state = torch.randn(6, 8, 5, 4)

    llrs = LlrReadout(8, num_bits=4, hidden_units=(12,))(state)
    channel = ChannelReadout(8, num_rx_antennas=2, hidden_units=(12,))(state)

    assert llrs.shape == (6, 4, 5, 4)
    assert channel.shape == (6, 4, 5, 4)
