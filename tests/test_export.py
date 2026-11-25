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
from neural_rx.export import export_onnx
from neural_rx.model import NeuralReceiver


def test_export_receiver_to_onnx(tmp_path: Path) -> None:
    config = ReceiverConfig(
        label="onnx-test",
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
    model = NeuralReceiver(config).eval()
    inputs = (
        torch.randn(1, 4, 3, 2),
        torch.randn(2, 4, 3, 2),
        torch.ones(1, 2),
        torch.randn(1, 2, 4, 3, 2),
    )
    destination = tmp_path / "receiver.onnx"

    export_onnx(model, inputs, destination)

    assert not destination.with_suffix(".onnx.data").exists()

    graph = onnx.load(destination)
    onnx.checker.check_model(graph)
    assert [output.name for output in graph.graph.output] == ["llrs"]


def test_export_pilotless_receiver_without_channel_input(tmp_path: Path) -> None:
    config = ReceiverConfig(
        label="onnx-e2e-test",
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
    model = NeuralReceiver(config).eval()
    inputs = (
        torch.randn(1, 4, 3, 2),
        torch.randn(1, 4, 3, 2),
        torch.ones(1, 1),
    )
    destination = tmp_path / "receiver-e2e.onnx"

    export_onnx(model, inputs, destination)

    graph = onnx.load(destination)
    onnx.checker.check_model(graph)
    assert [value.name for value in graph.graph.input] == [
        "received",
        "pilot_features",
        "active_users",
    ]


def test_export_batched_pilot_features_with_dynamic_batch(tmp_path: Path) -> None:
    config = ReceiverConfig(
        label="onnx-batched-pilots",
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
    inputs = (
        torch.randn(2, 4, 3, 2),
        torch.randn(2, 1, 4, 3, 2),
        torch.ones(2, 1),
    )
    destination = tmp_path / "receiver-batched-pilots.onnx"

    export_onnx(NeuralReceiver(config).eval(), inputs, destination)

    graph = onnx.load(destination)
    pilot_input = next(
        value for value in graph.graph.input if value.name == "pilot_features"
    )
    assert pilot_input.type.tensor_type.shape.dim[0].dim_param


def test_export_variable_mcs_uses_explicit_mask(tmp_path: Path) -> None:
    config = ReceiverConfig(
        label="onnx-variable-mcs",
        num_rx_antennas=1,
        num_bits_per_symbol=4,
        num_iterations=1,
        state_dim=4,
        init_units=(5,),
        aggregation_units=((5,),),
        update_units=((5,),),
        readout_units=(5,),
        convolution="conv",
        initial_channel_estimate=False,
        num_bits_per_symbol_by_mcs=(2, 4),
    )
    model = NeuralReceiver(config).eval()
    inputs = (
        torch.randn(1, 4, 3, 2),
        torch.randn(2, 4, 3, 2),
        torch.ones(1, 2),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
    )
    destination = tmp_path / "variable-mcs.onnx"

    export_onnx(model, inputs, destination)

    graph = onnx.load(destination)
    assert [value.name for value in graph.graph.input][-1] == "mcs_mask"
    feed = {
        value.name: tensor.numpy() for value, tensor in zip(graph.graph.input, inputs)
    }
    (actual,) = ReferenceEvaluator(graph).run(None, feed)
    with torch.no_grad():
        expected = model.forward_llrs(*inputs[:3], mcs_mask=inputs[3])
    torch.testing.assert_close(torch.from_numpy(actual), expected, rtol=1e-4, atol=1e-5)
