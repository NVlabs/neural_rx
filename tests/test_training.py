# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path

import pytest
import torch
from torch.nn import functional as functional

from neural_rx.config import ReceiverConfig
from neural_rx.model import NeuralReceiver, ReceiverOutput
from neural_rx.training import (
    apply_mcs_snr_offsets,
    load_checkpoint,
    receiver_loss,
    save_checkpoint,
)


def small_config() -> ReceiverConfig:
    return ReceiverConfig(
        label="checkpoint-test",
        num_rx_antennas=1,
        num_bits_per_symbol=2,
        num_iterations=1,
        state_dim=4,
        init_units=(5,),
        aggregation_units=((5,),),
        update_units=((5,),),
        readout_units=(5,),
        convolution="conv",
        initial_channel_estimate=False,
    )


def test_receiver_loss_ignores_inactive_users() -> None:
    active_users = torch.tensor([[1.0, 0.0]])
    logits = torch.tensor([[[[[2.0, -2.0]]], [[[50.0, -50.0]]]]])
    channel = torch.zeros(1, 2, 1, 1, 2)
    outputs = (ReceiverOutput(llrs=logits, channel=channel),)
    targets_a = torch.tensor([[[[[1.0, 0.0]]], [[[0.0, 1.0]]]]])
    targets_b = torch.tensor([[[[[1.0, 0.0]]], [[[1.0, 0.0]]]]])

    loss_a = receiver_loss(outputs, targets_a, active_users)
    loss_b = receiver_loss(outputs, targets_b, active_users)

    torch.testing.assert_close(loss_a.data, loss_b.data)
    expected = (
        functional.binary_cross_entropy_with_logits(
            logits[:, :1], targets_a[:, :1], reduction="mean"
        )
        / 2
    )
    torch.testing.assert_close(loss_a.data, expected)
    torch.testing.assert_close(loss_a.total, loss_a.data)


def test_receiver_loss_ignores_non_data_resource_elements() -> None:
    active_users = torch.ones(1, 1)
    logits = torch.tensor([[[[[2.0], [50.0]]]]])
    outputs = (ReceiverOutput(llrs=logits, channel=torch.zeros(1, 1, 1, 2, 2)),)
    targets_a = torch.tensor([[[[[1.0], [0.0]]]]])
    targets_b = torch.tensor([[[[[1.0], [1.0]]]]])
    data_mask = torch.tensor([[[True, False]]])

    loss_a = receiver_loss(outputs, targets_a, active_users, data_mask=data_mask)
    loss_b = receiver_loss(outputs, targets_b, active_users, data_mask=data_mask)

    torch.testing.assert_close(loss_a.data, loss_b.data)


def test_receiver_loss_ignores_bits_outside_selected_mcs() -> None:
    active_users = torch.ones(1, 2)
    logits = torch.tensor([[[[[2.0, -2.0, 50.0, -50.0]]], [[[2.0, -2.0, 2.0, -2.0]]]]])
    outputs = (ReceiverOutput(llrs=logits, channel=torch.zeros(1, 2, 1, 1, 2)),)
    targets_a = torch.tensor([[[[[1.0, 0.0, 0.0, 1.0]]], [[[1.0, 0.0, 1.0, 0.0]]]]])
    targets_b = targets_a.clone()
    targets_b[0, 0, ..., 2:] = 1 - targets_b[0, 0, ..., 2:]
    bit_mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]])

    loss_a = receiver_loss(outputs, targets_a, active_users, bit_mask=bit_mask)
    loss_b = receiver_loss(outputs, targets_b, active_users, bit_mask=bit_mask)

    torch.testing.assert_close(loss_a.data, loss_b.data)


def test_receiver_loss_averages_each_selected_codeword() -> None:
    active_users = torch.ones(1, 2)
    logits = torch.tensor([[[[[0.0, 0.0, 20.0, 20.0]]],
                            [[[0.0, 0.0, 0.0, 0.0]]]]])
    targets = torch.zeros_like(logits)
    outputs = (ReceiverOutput(llrs=logits, channel=torch.zeros(1, 2, 1, 1, 2)),)
    bit_mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0],
                              [1.0, 1.0, 1.0, 1.0]]])

    loss = receiver_loss(outputs, targets, active_users, bit_mask=bit_mask)

    first = functional.binary_cross_entropy_with_logits(
        logits[:, 0, ..., :2], targets[:, 0, ..., :2], reduction="mean")
    second = functional.binary_cross_entropy_with_logits(
        logits[:, 1], targets[:, 1], reduction="mean")
    torch.testing.assert_close(loss.data, (first + second) / 2)


def test_mcs_snr_offsets_sum_over_active_users() -> None:
    snr_db = torch.tensor([1.0, 2.0])
    mcs_mask = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [0.0, 1.0]],
        ]
    )
    active_users = torch.tensor([[1.0, 1.0], [1.0, 0.0]])

    adjusted = apply_mcs_snr_offsets(snr_db, mcs_mask, [0.0, 4.0], active_users)

    torch.testing.assert_close(adjusted, torch.tensor([5.0, 6.0]))


def test_receiver_loss_sums_iteration_and_channel_losses() -> None:
    active_users = torch.tensor([[1.0, 0.0]])
    logits = torch.zeros(1, 2, 1, 1, 1)
    channel = torch.zeros(1, 2, 1, 1, 2)
    outputs = (
        ReceiverOutput(llrs=logits, channel=channel),
        ReceiverOutput(llrs=logits, channel=channel),
    )
    channel_target = torch.full_like(channel, 2.0)

    loss = receiver_loss(
        outputs,
        torch.zeros_like(logits),
        active_users,
        channel_target=channel_target,
        channel_weight=0.25,
    )

    torch.testing.assert_close(loss.data, torch.log(torch.tensor(2.0)))
    torch.testing.assert_close(loss.channel, torch.tensor(4.0))
    torch.testing.assert_close(loss.total, loss.data + 0.25 * loss.channel)


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(7)
    config = small_config()
    source = NeuralReceiver(config)
    target = NeuralReceiver(config)
    checkpoint = tmp_path / "receiver.safetensors"

    save_checkpoint(source, checkpoint, config, metadata={"steps": 12})
    metadata = load_checkpoint(target, checkpoint)

    assert metadata["label"] == "checkpoint-test"
    assert metadata["steps"] == 12
    for source_value, target_value in zip(
        source.state_dict().values(), target.state_dict().values()
    ):
        torch.testing.assert_close(source_value, target_value)


def test_checkpoint_rejects_unexpected_extra_tensors(tmp_path: Path) -> None:
    config = small_config()
    checkpoint = tmp_path / "receiver.safetensors"
    save_checkpoint(
        NeuralReceiver(config),
        checkpoint,
        config,
        extra_tensors={"constellation": torch.ones(2)},
    )

    with pytest.raises(ValueError, match="extra tensors"):
        load_checkpoint(NeuralReceiver(config), checkpoint)
