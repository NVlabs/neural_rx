# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Aerial-compatible inference boundary for the PyTorch neural receiver."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from neural_rx.model import NeuralReceiver
from neural_rx.phy import SionnaPuschAdapter


class AerialReceiver(nn.Module):
    """Adapt Aerial slot tensors and sparse LS estimates to ``NeuralReceiver``.

    Inputs use separate real and imaginary tensors. ``received_*`` have shape
    ``[B,F,T,A]`` and ``ls_channel_*`` have shape ``[B,P,U,A]``. The latter are
    ordered by DMRS symbol, PRB, and in-PRB subcarrier, where
    ``P=D*(F/12)*K``. DMRS indices have shapes ``[U,D]`` and ``[U,K]``;
    ``active_users`` is ``[B,U]``.

    The output is a contiguous ``[B,Q,U,F,T]`` tensor. It uses the Aerial
    convention ``log(P(bit=0)/P(bit=1))``, the negative of the current
    receiver's training logits.
    """

    def __init__(
        self,
        receiver: NeuralReceiver,
        max_num_users: int,
        num_cdm_groups_without_data: int = 2,
    ) -> None:
        super().__init__()
        if max_num_users < 1:
            raise ValueError("max_num_users must be positive")
        if num_cdm_groups_without_data not in (1, 2):
            raise ValueError("Aerial export supports one or two CDM groups")
        self.receiver = receiver
        self.max_num_users = max_num_users
        self.num_cdm_groups_without_data = num_cdm_groups_without_data
        self.resource_elements_per_prb = 12

    @staticmethod
    def _normalize_distance(distance: Tensor) -> Tensor:
        distance = distance.to(torch.float32)
        return (distance - distance.mean(dim=-1, keepdim=True)) / distance.std(
            dim=-1, correction=0, keepdim=True
        ).clamp_min(1.0)

    def prepare_inputs(
        self,
        received_real: Tensor,
        received_imag: Tensor,
        ls_channel_real: Tensor,
        ls_channel_imag: Tensor,
        dmrs_ofdm_indices: Tensor,
        dmrs_subcarrier_indices: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return current-model received, pilot-feature, and full-channel tensors."""

        received = torch.cat((received_real, received_imag), dim=-1)
        sparse_channel = torch.cat((ls_channel_real, ls_channel_imag), dim=-1)
        batch, num_pilots, users, channel_components = sparse_channel.shape
        subcarriers, symbols = received.shape[1:3]
        dmrs_symbols = dmrs_ofdm_indices.shape[1]
        pilots_per_prb = dmrs_subcarrier_indices.shape[1]
        num_prbs = subcarriers // self.resource_elements_per_prb

        if not torch.compiler.is_compiling():
            if received_real.shape != received_imag.shape:
                raise ValueError("Received real and imaginary shapes must match")
            if ls_channel_real.shape != ls_channel_imag.shape:
                raise ValueError("LS-channel real and imaginary shapes must match")
            if users != self.max_num_users:
                raise ValueError(
                    "LS-channel user dimension does not match max_num_users"
                )
            if dmrs_ofdm_indices.shape[0] != users:
                raise ValueError("DMRS OFDM indices must have shape [U,D]")
            if dmrs_subcarrier_indices.shape[0] != users:
                raise ValueError("DMRS subcarrier indices must have shape [U,K]")
            if subcarriers % self.resource_elements_per_prb:
                raise ValueError(
                    "The subcarrier count must contain complete 12-RE PRBs"
                )
            if num_pilots != dmrs_symbols * num_prbs * pilots_per_prb:
                raise ValueError("Sparse LS pilot count must equal D*(F/12)*K")

        if self.num_cdm_groups_without_data == 2:
            # Aerial supplies the two FOCC estimates for a CDM pair consecutively.
            sparse_channel = (
                sparse_channel.reshape(batch, num_pilots // 2, 2, users, channel_components)
                .mean(dim=2)
                .repeat_interleave(2, dim=1)
            )
        sparse_channel = sparse_channel.reshape(
            batch,
            dmrs_symbols,
            num_prbs,
            pilots_per_prb,
            users,
            channel_components,
        ).permute(0, 4, 2, 1, 3, 5)
        sparse_channel = sparse_channel.flatten(3, 4)

        time = torch.arange(symbols, device=received.device)
        frequency = torch.arange(self.resource_elements_per_prb, device=received.device)
        time_delta = (time[None, :, None] - dmrs_ofdm_indices[:, None, :]).abs()
        frequency_delta = (
            frequency[None, :, None] - dmrs_subcarrier_indices[:, None, :]
        ).abs()
        distances = (
            time_delta[:, :, None, :, None] + frequency_delta[:, None, :, None, :]
        ).flatten(3, 4)
        nearest = distances.argmin(dim=-1).flatten(1, 2)
        nearest = nearest[None, :, None, :, None].expand(
            batch, users, num_prbs, -1, channel_components
        )
        full_channel = torch.gather(sparse_channel, 3, nearest)
        full_channel = full_channel.reshape(
            batch,
            users,
            num_prbs,
            symbols,
            self.resource_elements_per_prb,
            channel_components,
        ).permute(0, 1, 2, 4, 3, 5)
        full_channel = full_channel.flatten(2, 3)

        time_feature = self._normalize_distance(time_delta.amin(dim=-1))
        frequency_feature = self._normalize_distance(frequency_delta.amin(dim=-1))
        pilot_features = torch.stack(
            (
                time_feature[:, None, :].expand(-1, self.resource_elements_per_prb, -1),
                frequency_feature[:, :, None].expand(-1, -1, symbols),
            ),
            dim=-1,
        ).repeat(1, num_prbs, 1, 1)
        return received, pilot_features, full_channel

    def forward(
        self,
        received_real: Tensor,
        received_imag: Tensor,
        ls_channel_real: Tensor,
        ls_channel_imag: Tensor,
        active_users: Tensor,
        dmrs_ofdm_indices: Tensor,
        dmrs_subcarrier_indices: Tensor,
        mcs_mask: Tensor | None = None,
    ) -> Tensor:
        received, pilot_features, full_channel = self.prepare_inputs(
            received_real,
            received_imag,
            ls_channel_real,
            ls_channel_imag,
            dmrs_ofdm_indices,
            dmrs_subcarrier_indices,
        )
        llrs = self.receiver.forward_llrs(
            received,
            pilot_features,
            active_users,
            full_channel,
            mcs_mask=mcs_mask,
        )
        return -llrs.permute(0, 4, 1, 2, 3).contiguous()


def aerial_example_inputs(
    adapter: SionnaPuschAdapter, batch_size: int = 1
) -> tuple[Tensor, ...]:
    """Build export inputs from the configured PUSCH grid and DMRS pattern."""

    pilot_mask = adapter.pilot_mask
    dmrs_symbols = []
    dmrs_subcarriers = []
    for user_mask in pilot_mask:
        dmrs_symbols.append(torch.nonzero(user_mask.any(dim=0)).flatten())
        residues = torch.nonzero(user_mask.any(dim=1)[:12]).flatten()
        dmrs_subcarriers.append(residues)
    dmrs_ofdm_indices = torch.stack(dmrs_symbols).to(torch.int32)
    dmrs_subcarrier_indices = torch.stack(dmrs_subcarriers).to(torch.int32)
    subcarriers = adapter.resource_grid.fft_size
    symbols = adapter.resource_grid.num_ofdm_symbols
    antennas = adapter.num_rx_antennas
    num_pilots = (
        dmrs_ofdm_indices.shape[1]
        * (subcarriers // 12)
        * dmrs_subcarrier_indices.shape[1]
    )
    shape = (batch_size, subcarriers, symbols, antennas)
    channel_shape = (batch_size, num_pilots, adapter.num_users, antennas)
    inputs = (
        torch.randn(shape, device=adapter.device),
        torch.randn(shape, device=adapter.device),
        torch.randn(channel_shape, device=adapter.device),
        torch.randn(channel_shape, device=adapter.device),
        torch.ones((batch_size, adapter.num_users), device=adapter.device),
        dmrs_ofdm_indices.to(adapter.device),
        dmrs_subcarrier_indices.to(adapter.device),
    )
    if len(adapter.num_bits_by_mcs) > 1:
        mcs_mask = torch.zeros(
            batch_size,
            adapter.num_users,
            len(adapter.num_bits_by_mcs),
            device=adapter.device,
        )
        mcs_mask[..., 0] = 1
        inputs = (*inputs, mcs_mask)
    return inputs


def export_aerial_onnx(
    wrapper: AerialReceiver,
    inputs: tuple[Tensor, ...],
    destination: str | Path,
) -> None:
    """Export the fixed-grid Aerial boundary with a dynamic batch dimension."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    batch = torch.export.Dim("batch", min=1)
    variable_mcs = len(wrapper.receiver.modulation_orders) > 1
    expected_inputs = 8 if variable_mcs else 7
    if len(inputs) != expected_inputs:
        requirement = " including mcs_mask" if variable_mcs else ""
        raise ValueError(f"Expected {expected_inputs} Aerial ONNX inputs{requirement}")
    dynamic_shapes = (
        {0: batch},
        {0: batch},
        {0: batch},
        {0: batch},
        {0: batch},
        None,
        None,
    )
    input_names = [
        "received_real",
        "received_imag",
        "ls_channel_real",
        "ls_channel_imag",
        "active_users",
        "dmrs_ofdm_indices",
        "dmrs_subcarrier_indices",
    ]
    if variable_mcs:
        dynamic_shapes = (*dynamic_shapes, {0: batch})
        input_names.append("mcs_mask")
    with torch.no_grad():
        torch.onnx.export(
            wrapper.eval(),
            inputs,
            destination,
            input_names=input_names,
            output_names=["llrs"],
            dynamic_shapes=dynamic_shapes,
            opset_version=18,
            dynamo=True,
            external_data=False,
        )
