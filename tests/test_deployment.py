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

import onnx
import torch
from onnx.reference import ReferenceEvaluator

from neural_rx.config import ReceiverConfig
from neural_rx.deployment import AerialReceiver, export_aerial_onnx
from neural_rx.model import NeuralReceiver


def _wrapper() -> AerialReceiver:
    config = ReceiverConfig(
        label="aerial-test",
        num_rx_antennas=1,
        num_bits_per_symbol=2,
        num_iterations=1,
        state_dim=4,
        init_units=(5,),
        aggregation_units=((5,),),
        update_units=((5,),),
        readout_units=(5,),
        convolution="conv",
        initial_channel_estimate=True,
    )
    return AerialReceiver(NeuralReceiver(config).eval(), max_num_users=2).eval()


def _inputs(batch: int = 2) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(19)
    subcarriers = 24
    symbols = 4
    users = 2
    dmrs_ofdm = torch.tensor([[0, 3], [0, 3]], dtype=torch.int32)
    dmrs_subcarriers = torch.tensor(
        [[0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11]], dtype=torch.int32
    )
    num_pilots = 2 * (subcarriers // 12) * 6
    return (
        torch.randn(batch, subcarriers, symbols, 1),
        torch.randn(batch, subcarriers, symbols, 1),
        torch.randn(batch, num_pilots, users, 1),
        torch.randn(batch, num_pilots, users, 1),
        torch.tensor([[1.0, 1.0]] * batch),
        dmrs_ofdm,
        dmrs_subcarriers,
    )


def test_aerial_wrapper_matches_current_receiver_boundary() -> None:
    wrapper = _wrapper()
    inputs = _inputs()

    received, pilots, channel = wrapper.prepare_inputs(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[5], inputs[6]
    )
    with torch.no_grad():
        expected = wrapper.receiver.forward_llrs(received, pilots, inputs[4], channel)
        actual = wrapper(*inputs)

    assert received.shape == (2, 24, 4, 2)
    assert pilots.shape == (2, 24, 4, 2)
    assert channel.shape == (2, 2, 24, 4, 2)
    assert actual.shape == (2, 2, 2, 24, 4)
    torch.testing.assert_close(actual, -expected.permute(0, 4, 1, 2, 3))


def test_sparse_channel_is_interpolated_per_prb_after_focc_removal() -> None:
    wrapper = _wrapper()
    inputs = list(_inputs(batch=1))
    # Each adjacent FOCC pair averages to 0.5, 2.5, ..., 22.5.
    values = torch.arange(24, dtype=torch.float32).reshape(1, 24, 1, 1)
    inputs[2] = values.expand(-1, -1, 2, -1)
    inputs[3] = torch.zeros_like(inputs[2])

    _, _, channel = wrapper.prepare_inputs(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[5], inputs[6]
    )

    torch.testing.assert_close(channel[0, 0, 0, 0, 0], torch.tensor(0.5))
    torch.testing.assert_close(channel[0, 0, 0, 3, 0], torch.tensor(12.5))
    torch.testing.assert_close(channel[0, 0, 12, 0, 0], torch.tensor(6.5))
    assert torch.count_nonzero(channel[..., 1]) == 0


def test_single_cdm_group_keeps_each_sparse_channel_estimate() -> None:
    wrapper = _wrapper()
    wrapper.num_cdm_groups_without_data = 1
    inputs = list(_inputs(batch=1))
    values = torch.arange(24, dtype=torch.float32).reshape(1, 24, 1, 1)
    inputs[2] = values.expand(-1, -1, 2, -1)
    inputs[3] = torch.zeros_like(inputs[2])

    _, _, channel = wrapper.prepare_inputs(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[5], inputs[6]
    )

    torch.testing.assert_close(channel[0, 0, 0, 0, 0], torch.tensor(0.0))
    torch.testing.assert_close(channel[0, 0, 0, 3, 0], torch.tensor(12.0))
    torch.testing.assert_close(channel[0, 0, 12, 0, 0], torch.tensor(6.0))


def test_export_aerial_receiver_to_onnx(tmp_path: Path) -> None:
    destination = tmp_path / "aerial-receiver.onnx"
    wrapper = _wrapper()
    inputs = _inputs(batch=1)

    export_aerial_onnx(wrapper, inputs, destination)

    assert not destination.with_suffix(".onnx.data").exists()

    graph = onnx.load(destination)
    onnx.checker.check_model(graph)
    assert [value.name for value in graph.graph.input] == [
        "received_real",
        "received_imag",
        "ls_channel_real",
        "ls_channel_imag",
        "active_users",
        "dmrs_ofdm_indices",
        "dmrs_subcarrier_indices",
    ]
    output = graph.graph.output[0]
    assert output.name == "llrs"
    dimensions = output.type.tensor_type.shape.dim
    assert dimensions[0].dim_param
    assert [dimension.dim_value for dimension in dimensions[1:]] == [2, 2, 24, 4]

    ort_inputs = {
        value.name: tensor.detach().numpy()
        for value, tensor in zip(graph.graph.input, inputs)
    }
    (onnx_llrs,) = ReferenceEvaluator(graph).run(None, ort_inputs)
    with torch.no_grad():
        expected = wrapper(*inputs)
    torch.testing.assert_close(torch.from_numpy(onnx_llrs), expected, rtol=1e-4, atol=1e-5)
