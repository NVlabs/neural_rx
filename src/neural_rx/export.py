# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""ONNX export for the framework-independent receiver boundary."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from neural_rx.model import NeuralReceiver


class _OnnxReceiver(nn.Module):
    def __init__(self, receiver: NeuralReceiver) -> None:
        super().__init__()
        self.receiver = receiver

    def forward(
        self,
        received: Tensor,
        pilot_features: Tensor,
        active_users: Tensor,
        initial_channel: Tensor,
    ) -> Tensor:
        return self.receiver.forward_llrs(
            received,
            pilot_features,
            active_users,
            initial_channel,
        )


class _OnnxVariableMcsReceiver(nn.Module):
    def __init__(self, receiver: NeuralReceiver) -> None:
        super().__init__()
        self.receiver = receiver

    def forward(
        self,
        received: Tensor,
        pilot_features: Tensor,
        active_users: Tensor,
        initial_channel: Tensor,
        mcs_mask: Tensor,
    ) -> Tensor:
        return self.receiver.forward_llrs(
            received,
            pilot_features,
            active_users,
            initial_channel,
            mcs_mask=mcs_mask,
        )


class _OnnxPilotlessReceiver(nn.Module):
    def __init__(self, receiver: NeuralReceiver) -> None:
        super().__init__()
        self.receiver = receiver

    def forward(
        self,
        received: Tensor,
        pilot_features: Tensor,
        active_users: Tensor,
    ) -> Tensor:
        return self.receiver.forward_llrs(received, pilot_features, active_users)


class _OnnxVariableMcsPilotlessReceiver(nn.Module):
    def __init__(self, receiver: NeuralReceiver) -> None:
        super().__init__()
        self.receiver = receiver

    def forward(
        self,
        received: Tensor,
        pilot_features: Tensor,
        active_users: Tensor,
        mcs_mask: Tensor,
    ) -> Tensor:
        return self.receiver.forward_llrs(
            received,
            pilot_features,
            active_users,
            mcs_mask=mcs_mask,
        )


def export_onnx(
    receiver: NeuralReceiver,
    inputs: tuple[Tensor, ...],
    destination: str | Path,
) -> None:
    """Export a receiver with dynamic batch size and fixed radio-grid dimensions."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    receiver = receiver.eval()
    batch = torch.export.Dim("batch", min=1)
    pilot_batch = {0: batch} if inputs[1].ndim == 5 else None
    variable_mcs = len(receiver.modulation_orders) > 1
    expected_inputs = (
        3 + int(receiver.config.initial_channel_estimate) + int(variable_mcs)
    )
    if len(inputs) != expected_inputs:
        requirement = " including mcs_mask" if variable_mcs else ""
        raise ValueError(f"Expected {expected_inputs} ONNX inputs{requirement}")
    if receiver.config.initial_channel_estimate:
        wrapper = (
            _OnnxVariableMcsReceiver(receiver)
            if variable_mcs
            else _OnnxReceiver(receiver)
        ).eval()
        input_names = [
            "received",
            "pilot_features",
            "active_users",
            "initial_channel",
        ]
        dynamic_shapes = ({0: batch}, pilot_batch, {0: batch}, {0: batch})
        if variable_mcs:
            input_names.append("mcs_mask")
            dynamic_shapes = (*dynamic_shapes, {0: batch})
    else:
        wrapper = (
            _OnnxVariableMcsPilotlessReceiver(receiver)
            if variable_mcs
            else _OnnxPilotlessReceiver(receiver)
        ).eval()
        input_names = ["received", "pilot_features", "active_users"]
        dynamic_shapes = ({0: batch}, pilot_batch, {0: batch})
        if variable_mcs:
            input_names.append("mcs_mask")
            dynamic_shapes = (*dynamic_shapes, {0: batch})
    output_names = ["llrs"]
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            inputs,
            destination,
            input_names=input_names,
            output_names=output_names,
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
