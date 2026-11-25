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

from neural_rx.config import ReceiverConfig
from neural_rx.model import NeuralReceiver


def small_config() -> ReceiverConfig:
    return ReceiverConfig(
        label="test",
        num_rx_antennas=2,
        num_bits_per_symbol=4,
        num_iterations=2,
        state_dim=6,
        init_units=(8,),
        aggregation_units=((7,), (7,)),
        update_units=((8,), (8,)),
        readout_units=(8,),
        convolution="sepconv",
        initial_channel_estimate=True,
    )


def receiver_inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(3)
    received = torch.randn(2, 5, 4, 4)
    pilots = torch.randn(3, 5, 4, 2)
    active_users = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    initial_channel = torch.randn(2, 3, 5, 4, 4)
    return received, pilots, active_users, initial_channel


def test_receiver_returns_only_final_iteration_by_default() -> None:
    model = NeuralReceiver(small_config())

    outputs = model(*receiver_inputs())

    assert len(outputs) == 1
    assert outputs[0].llrs.shape == (2, 3, 5, 4, 4)
    assert outputs[0].channel.shape == (2, 3, 5, 4, 4)


def test_receiver_llr_inference_skips_channel_readout() -> None:
    model = NeuralReceiver(small_config()).eval()
    inputs = receiver_inputs()
    with torch.no_grad():
        expected = model(*inputs)[0].llrs
    channel_readout_calls = []
    handle = model.channel_readout.register_forward_hook(
        lambda *_: channel_readout_calls.append(True)
    )

    with torch.no_grad():
        llrs = model.forward_llrs(*inputs)
    handle.remove()

    torch.testing.assert_close(llrs, expected)
    assert channel_readout_calls == []


def test_receiver_can_return_every_unshared_iteration() -> None:
    model = NeuralReceiver(small_config())

    outputs = model(*receiver_inputs(), return_all_iterations=True)

    assert len(outputs) == 2
    assert model.iterations[0] is not model.iterations[1]


def test_receiver_masks_inactive_user_outputs() -> None:
    model = NeuralReceiver(small_config())

    output = model(*receiver_inputs())[0]

    torch.testing.assert_close(output.llrs[0, 2], torch.zeros_like(output.llrs[0, 2]))
    torch.testing.assert_close(output.channel[1, 1:], torch.zeros_like(output.channel[1, 1:]))


def test_receiver_backward_reaches_all_iterations() -> None:
    model = NeuralReceiver(small_config())
    outputs = model(*receiver_inputs(), return_all_iterations=True)

    loss = sum(output.llrs.square().mean() for output in outputs)
    loss = loss + sum(output.channel.square().mean() for output in outputs)
    loss.backward()

    assert all(parameter.grad is not None for parameter in model.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_receiver_handles_zero_received_power() -> None:
    model = NeuralReceiver(small_config())
    received, pilots, active_users, initial_channel = receiver_inputs()

    output = model(torch.zeros_like(received), pilots, active_users, initial_channel)[0]

    assert torch.isfinite(output.llrs).all()
    assert torch.isfinite(output.channel).all()


def test_receiver_routes_users_through_mcs_specific_io() -> None:
    config = small_config()
    config = ReceiverConfig(
        **{
            **config.__dict__,
            "num_bits_per_symbol_by_mcs": (2, 4),
            "mcs_indices": (9, 14),
        }
    )
    model = NeuralReceiver(config)
    received, pilots, active_users, initial_channel = receiver_inputs()
    mcs_mask = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]] * received.shape[0]
    )

    output = model(
        received,
        pilots,
        active_users,
        initial_channel,
        mcs_mask=mcs_mask,
    )[0]

    assert len(model.mcs_initializers) == 1
    assert len(model.mcs_llr_readouts) == 1
    assert torch.count_nonzero(output.llrs[:, (0, 2), ..., 2:]) == 0


def test_receiver_max_llr_mode_masks_bits_without_mcs_specific_io() -> None:
    config = small_config()
    config = ReceiverConfig(
        **{
            **config.__dict__,
            "num_bits_per_symbol_by_mcs": (2, 4),
            "mcs_indices": (9, 14),
            "mcs_masking": True,
        }
    )
    model = NeuralReceiver(config)
    received, pilots, active_users, initial_channel = receiver_inputs()
    mcs_mask = torch.zeros(received.shape[0], active_users.shape[1], 2)
    mcs_mask[..., 0] = 1

    output = model(
        received,
        pilots,
        active_users,
        initial_channel,
        mcs_mask=mcs_mask,
    )[0]

    assert len(model.mcs_initializers) == 0
    assert len(model.mcs_llr_readouts) == 0
    assert torch.count_nonzero(output.llrs[..., 2:]) == 0


def test_inactive_users_do_not_influence_active_users() -> None:
    model = NeuralReceiver(small_config()).eval()
    received, pilots, active_users, initial_channel = receiver_inputs()
    changed = initial_channel.clone()
    changed[0, 2] += 10.0  # user 2 is inactive in the first batch entry

    with torch.no_grad():
        expected = model(received, pilots, active_users, initial_channel)[0]
        output = model(received, pilots, active_users, changed)[0]

    torch.testing.assert_close(output.llrs, expected.llrs)
    torch.testing.assert_close(output.channel, expected.channel)


def test_aggregation_averages_other_active_users() -> None:
    from neural_rx.model import _mixing_weights

    weights = _mixing_weights(torch.tensor([[1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0]]))

    torch.testing.assert_close(weights[0, 0], torch.tensor([0.0, 0.5, 0.0, 0.5]))
    torch.testing.assert_close(weights[0, 2], torch.tensor([1 / 3, 1 / 3, 0.0, 1 / 3]))
    torch.testing.assert_close(weights[1, 0], torch.zeros(4))
    torch.testing.assert_close(weights[1, 1], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_single_user_receiver_skips_aggregation() -> None:
    model = NeuralReceiver(small_config()).eval()
    received, pilots, _, initial_channel = receiver_inputs()
    calls = []
    handle = model.iterations[0].aggregator.output.register_forward_hook(
        lambda *_: calls.append(True)
    )

    with torch.no_grad():
        output = model(received, pilots[:1], torch.ones(2, 1), initial_channel[:, :1])[0]
    handle.remove()

    assert calls == []
    assert torch.isfinite(output.llrs).all()


def test_feature_entry_point_matches_forward() -> None:
    model = NeuralReceiver(small_config()).eval()
    received, pilots, active_users, initial_channel = receiver_inputs()

    with torch.no_grad():
        expected = model(received, pilots, active_users, initial_channel)[0]
        scale = model.input_scale(received)
        features, positions = model.initializer_features(
            received * scale, pilots, initial_channel * scale.unsqueeze(1)
        )
        output = model.forward_features(features, positions, active_users)
        llrs_only = model.forward_features(
            features, positions, active_users, with_channel=False
        )

    torch.testing.assert_close(output.llrs, expected.llrs)
    torch.testing.assert_close(output.channel, expected.channel)
    torch.testing.assert_close(llrs_only.llrs, expected.llrs)
    assert llrs_only.channel is None
