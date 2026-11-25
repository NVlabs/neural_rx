# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""PyTorch implementation of the convolutional graph neural receiver."""

from __future__ import annotations

from collections.abc import Iterable
from typing import NamedTuple

import torch
from torch import Tensor, nn

from neural_rx.config import ReceiverConfig


class SeparableConv2d(nn.Module):
    """Depthwise 3x3 convolution followed by pointwise channel mixing."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.pointwise(self.depthwise(inputs))


def _convolution(kind: str, in_channels: int, out_channels: int) -> nn.Module:
    if kind == "sepconv":
        return SeparableConv2d(in_channels, out_channels)
    if kind == "conv":
        return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    raise ValueError(f"Unknown convolution type: {kind!r}")


def _broadcast_pilot_features(features: Tensor, batch: int, users: int) -> Tensor:
    if features.ndim == 4:
        if features.shape[0] != users:
            raise ValueError("Pilot feature user dimension does not match receiver input")
        return features.unsqueeze(0).expand(batch, -1, -1, -1, -1)
    if features.ndim == 5:
        if features.shape[:2] != (batch, users):
            raise ValueError("Batched pilot feature dimensions do not match receiver input")
        return features
    raise ValueError("Pilot features must have shape [U,F,T,2] or [B,U,F,T,2]")


def _channels_to_grid(inputs: Tensor, batch: int, users: int) -> Tensor:
    _, channels, subcarriers, symbols = inputs.shape
    return inputs.reshape(batch, users, channels, subcarriers, symbols).permute(0, 1, 3, 4, 2)


def _mixing_weights(active_users: Tensor) -> Tensor:
    """Return ``[B,U,U]`` weights averaging the *other* active users."""

    users = active_users.shape[1]
    others = 1.0 - torch.eye(users, dtype=active_users.dtype, device=active_users.device)
    weights = others * active_users[:, None, :]
    return weights / weights.sum(dim=2, keepdim=True).clamp_min(1.0)


class StateInitializer(nn.Module):
    """Create one channel-first state grid per user."""

    def __init__(
        self,
        num_rx_antennas: int,
        state_dim: int,
        hidden_units: Iterable[int],
        convolution: str,
        initial_channel_estimate: bool,
    ) -> None:
        super().__init__()
        input_channels = 2 * num_rx_antennas + 2
        if initial_channel_estimate:
            input_channels += 2 * num_rx_antennas
        self.initial_channel_estimate = initial_channel_estimate
        self.hidden = nn.ModuleList()
        for units in hidden_units:
            self.hidden.append(_convolution(convolution, input_channels, units))
            input_channels = units
        self.output = _convolution(convolution, input_channels, state_dim)

    def assemble(
        self,
        received: Tensor,
        pilot_features: Tensor,
        initial_channel: Tensor | None,
    ) -> Tensor:
        """Stack the per-user network input ``[B*U,C,F,T]`` without arithmetic."""

        if received.ndim != 4:
            raise ValueError("Received input must have shape [B,F,T,2*A]")
        batch, subcarriers, symbols, _ = received.shape
        if pilot_features.ndim != 5 or pilot_features.shape[0] != batch:
            raise ValueError("Pilot features must have shape [B,U,2,F,T]")
        _, users, _, pilot_subcarriers, pilot_symbols = pilot_features.shape
        if (pilot_subcarriers, pilot_symbols) != (subcarriers, symbols):
            raise ValueError("Pilot feature grid dimensions do not match receiver input")
        received_channels = received.permute(0, 3, 1, 2).unsqueeze(1)
        received_per_user = received_channels.expand(-1, users, -1, -1, -1)
        pieces = [received_per_user, pilot_features]
        if self.initial_channel_estimate:
            if initial_channel is None:
                raise ValueError("This receiver expects an initial channel estimate")
            if initial_channel.shape[:4] != (batch, users, subcarriers, symbols):
                raise ValueError("Initial channel dimensions do not match receiver input")
            pieces.append(initial_channel.permute(0, 1, 4, 2, 3))
        return torch.cat(pieces, dim=2).reshape(batch * users, -1, subcarriers, symbols)

    def from_features(self, features: Tensor) -> Tensor:
        state = features
        for layer in self.hidden:
            state = torch.relu(layer(state))
        return self.output(state)

    def forward(
        self,
        received: Tensor,
        pilot_features: Tensor,
        initial_channel: Tensor | None,
    ) -> Tensor:
        return self.from_features(self.assemble(received, pilot_features, initial_channel))


class UserStateAggregator(nn.Module):
    """Transform states and average the other active users for every user."""

    def __init__(self, state_dim: int, hidden_units: Iterable[int]) -> None:
        super().__init__()
        input_dim = state_dim
        self.hidden = nn.ModuleList()
        for units in hidden_units:
            self.hidden.append(nn.Conv2d(input_dim, units, kernel_size=1))
            input_dim = units
        self.output = nn.Conv2d(input_dim, state_dim, kernel_size=1)

    def forward(
        self, state: Tensor, active_users: Tensor, mixing: Tensor | None = None
    ) -> Tensor:
        """Aggregate ``[B*U,C,F,T]`` states.

        ``mixing`` are the precomputed ``[B,U,U]`` weights of
        :func:`_mixing_weights`. The user average is one batched matrix
        product: inactive users have zero weight, so neither the states nor
        the messages need to be masked.
        """

        batch, users = active_users.shape
        if state.shape[0] != batch * users:
            raise ValueError("State and active-user dimensions do not match")
        if users == 1:
            # A single user never receives a message.
            return torch.zeros_like(state)
        transformed = state
        for layer in self.hidden:
            transformed = torch.relu(layer(transformed))
        transformed = self.output(transformed)
        if mixing is None:
            mixing = _mixing_weights(active_users.to(dtype=state.dtype))
        _, channels, subcarriers, symbols = transformed.shape
        aggregated = torch.matmul(
            mixing.to(dtype=transformed.dtype), transformed.reshape(batch, users, -1)
        )
        return aggregated.reshape(batch * users, channels, subcarriers, symbols)


class StateUpdater(nn.Module):
    """Update per-user states with aggregated interference information."""

    def __init__(self, state_dim: int, hidden_units: Iterable[int], convolution: str) -> None:
        super().__init__()
        input_channels = 2 * state_dim + 2
        self.hidden = nn.ModuleList()
        for units in hidden_units:
            self.hidden.append(_convolution(convolution, input_channels, units))
            input_channels = units
        self.output = _convolution(convolution, input_channels, state_dim)

    def forward(self, state: Tensor, aggregated: Tensor, pilot_features: Tensor) -> Tensor:
        update = torch.cat([aggregated, state, pilot_features], dim=1)
        for layer in self.hidden:
            update = torch.relu(layer(update))
        update = self.output(update)
        return state + update


class _Readout(nn.Module):
    def __init__(self, state_dim: int, output_dim: int, hidden_units: Iterable[int]) -> None:
        super().__init__()
        input_dim = state_dim
        self.hidden = nn.ModuleList()
        for units in hidden_units:
            self.hidden.append(nn.Conv2d(input_dim, units, kernel_size=1))
            input_dim = units
        self.output = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, state: Tensor) -> Tensor:
        output = state
        for layer in self.hidden:
            output = torch.relu(layer(output))
        return self.output(output)


class LlrReadout(_Readout):
    def __init__(self, state_dim: int, num_bits: int, hidden_units: Iterable[int]) -> None:
        super().__init__(state_dim, num_bits, hidden_units)


class ChannelReadout(_Readout):
    def __init__(self, state_dim: int, num_rx_antennas: int, hidden_units: Iterable[int]) -> None:
        super().__init__(state_dim, 2 * num_rx_antennas, hidden_units)


class ReceiverOutput(NamedTuple):
    """Readouts produced by one receiver iteration."""

    llrs: Tensor
    channel: Tensor | None


class _ReceiverIteration(nn.Module):
    def __init__(
        self,
        state_dim: int,
        aggregation_units: Iterable[int],
        update_units: Iterable[int],
        convolution: str,
    ) -> None:
        super().__init__()
        self.aggregator = UserStateAggregator(state_dim, aggregation_units)
        self.updater = StateUpdater(state_dim, update_units, convolution)

    def forward(
        self,
        state: Tensor,
        pilots: Tensor,
        active_users: Tensor,
        mixing: Tensor | None = None,
    ) -> Tensor:
        aggregated = self.aggregator(state, active_users, mixing)
        return self.updater(state, aggregated, pilots)


class _PreparedState(NamedTuple):
    state: Tensor
    pilots: Tensor
    mixing: Tensor
    selected_mcs: Tensor
    llr_mask: Tensor  # zero for inactive users and bits unused by the selected MCS
    channel_mask: Tensor  # zero for inactive users


class NeuralReceiver(nn.Module):
    """Iterative CGNN receiver independent of the surrounding PHY simulator.

    The receiver operates on ``[B*U,C,F,T]`` state grids. Inactive users are
    handled by zero aggregation weights and by masking the readouts. Their
    intermediate states are never observed and therefore not masked, which
    removes all full-resolution elementwise operations from the inference
    graph.
    """

    def __init__(self, config: ReceiverConfig) -> None:
        super().__init__()
        self.config = config
        modulation_orders = config.num_bits_per_symbol_by_mcs or (
            config.num_bits_per_symbol,
        )
        self.modulation_orders = modulation_orders
        self.initializer = StateInitializer(
            num_rx_antennas=config.num_rx_antennas,
            state_dim=config.state_dim,
            hidden_units=config.init_units,
            convolution=config.convolution,
            initial_channel_estimate=config.initial_channel_estimate,
        )
        self.mcs_initializers = nn.ModuleList()
        if len(modulation_orders) > 1 and not config.mcs_masking:
            self.mcs_initializers.extend(
                StateInitializer(
                    num_rx_antennas=config.num_rx_antennas,
                    state_dim=config.state_dim,
                    hidden_units=config.init_units,
                    convolution=config.convolution,
                    initial_channel_estimate=config.initial_channel_estimate,
                )
                for _ in modulation_orders[1:]
            )
        self.iterations = nn.ModuleList(
            _ReceiverIteration(
                config.state_dim,
                config.aggregation_units[index],
                config.update_units[index],
                config.convolution,
            )
            for index in range(config.num_iterations)
        )
        primary_bits = (
            modulation_orders[0]
            if len(modulation_orders) > 1 and not config.mcs_masking
            else config.num_bits_per_symbol
        )
        self.llr_readout = LlrReadout(config.state_dim, primary_bits, config.readout_units)
        self.mcs_llr_readouts = nn.ModuleList()
        if len(modulation_orders) > 1 and not config.mcs_masking:
            self.mcs_llr_readouts.extend(
                LlrReadout(config.state_dim, num_bits, config.readout_units)
                for num_bits in modulation_orders[1:]
            )
        self.channel_readout = ChannelReadout(
            config.state_dim, config.num_rx_antennas, config.readout_units
        )

    def forward(
        self,
        received: Tensor,
        pilot_features: Tensor,
        active_users: Tensor,
        initial_channel: Tensor | None = None,
        *,
        mcs_mask: Tensor | None = None,
        return_all_iterations: bool = False,
    ) -> tuple[ReceiverOutput, ...]:
        prepared = self._prepare_state(
            received, pilot_features, active_users, initial_channel, mcs_mask
        )
        return self._iterate(
            prepared, active_users, return_all_iterations=return_all_iterations
        )

    def forward_llrs(
        self,
        received: Tensor,
        pilot_features: Tensor,
        active_users: Tensor,
        initial_channel: Tensor | None = None,
        *,
        mcs_mask: Tensor | None = None,
    ) -> Tensor:
        """Run inference without evaluating the training-only channel readout."""

        prepared = self._prepare_state(
            received, pilot_features, active_users, initial_channel, mcs_mask
        )
        return self._iterate(prepared, active_users, with_channel=False)[-1].llrs

    # Deployment entry points. Exports with a fixed resource grid prepare the
    # network inputs themselves (see TrainingAlignedAerialReceiver) and then
    # call forward_features. Training and evaluation only use forward and
    # forward_llrs.

    @staticmethod
    def input_scale(received: Tensor) -> Tensor:
        """Return the ``[B,1,1,1]`` factor normalizing the received power."""

        power = received.square().mean(dim=(1, 2, 3), keepdim=True).sqrt()
        epsilon = torch.finfo(received.dtype).eps
        return torch.where(power > epsilon, power.reciprocal(), torch.ones_like(power))

    def initializer_features(
        self,
        received: Tensor,
        pilot_features: Tensor,
        initial_channel: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Rearrange receiver inputs into network inputs without arithmetic.

        Returns the initializer input ``[B*U,C,F,T]`` and the positional
        features ``[B*U,2,F,T]`` consumed by every state update.
        """

        batch, subcarriers, symbols, _ = received.shape
        if pilot_features.ndim == 4:
            users = pilot_features.shape[0]
        else:
            users = pilot_features.shape[1]
        pilots = _broadcast_pilot_features(pilot_features, batch, users).permute(
            0, 1, 4, 2, 3
        )
        features = self.initializer.assemble(received, pilots, initial_channel)
        return features, pilots.reshape(batch * users, 2, subcarriers, symbols)

    def forward_features(
        self,
        features: Tensor,
        pilots: Tensor,
        active_users: Tensor,
        *,
        mcs_mask: Tensor | None = None,
        with_channel: bool = True,
    ) -> ReceiverOutput:
        """Run the receiver on inputs prepared by :meth:`initializer_features`.

        Deployment boundaries with a fixed resource grid use this entry point
        to replace the generic input rearrangement by one precomputed gather.
        ``features`` must already be scaled by :meth:`input_scale`.
        """

        prepared = self._state_from_features(features, pilots, active_users, mcs_mask)
        return self._iterate(prepared, active_users, with_channel=with_channel)[-1]

    def _iterate(
        self,
        prepared: _PreparedState,
        active_users: Tensor,
        *,
        return_all_iterations: bool = False,
        with_channel: bool = True,
    ) -> tuple[ReceiverOutput, ...]:
        state = prepared.state
        batch, users = active_users.shape
        outputs: list[ReceiverOutput] = []
        for index, iteration in enumerate(self.iterations):
            state = iteration(state, prepared.pilots, active_users, prepared.mixing)
            if return_all_iterations or index == len(self.iterations) - 1:
                llrs, channel = self._readout(state, prepared, with_channel=with_channel)
                outputs.append(
                    ReceiverOutput(
                        llrs=_channels_to_grid(llrs, batch, users),
                        channel=(
                            None
                            if channel is None
                            else _channels_to_grid(channel, batch, users)
                        ),
                    )
                )
        return tuple(outputs)

    def _readout(
        self, state: Tensor, prepared: _PreparedState, *, with_channel: bool
    ) -> tuple[Tensor, Tensor | None]:
        """Return ``[B*U,Q,F,T]`` LLRs and optionally the channel estimate.

        Both are zeroed for inactive users; LLRs also for unused bits.
        """

        primary = self.llr_readout(state)
        channel = None
        if with_channel:
            channel = self.channel_readout(state) * prepared.channel_mask
        if len(self.modulation_orders) == 1 or self.config.mcs_masking:
            llrs = primary
        else:
            num_bits = self.config.num_bits_per_symbol
            readouts = (self.llr_readout, *self.mcs_llr_readouts)
            llrs = None
            for index, readout in enumerate(readouts):
                value = primary if index == 0 else readout(state)
                value = torch.nn.functional.pad(
                    value, (0, 0, 0, 0, 0, num_bits - value.shape[1])
                )
                value = value * prepared.selected_mcs[:, index, None, None, None]
                llrs = value if llrs is None else llrs + value
        return llrs * prepared.llr_mask, channel

    def _prepare_state(
        self,
        received: Tensor,
        pilot_features: Tensor,
        active_users: Tensor,
        initial_channel: Tensor | None,
        mcs_mask: Tensor | None,
    ) -> _PreparedState:
        if active_users.ndim != 2:
            raise ValueError("Active users must have shape [B,U]")
        batch, users = active_users.shape
        if received.shape[0] != batch:
            raise ValueError("Received input and active-user batch dimensions differ")
        if pilot_features.shape[-4] != users:
            raise ValueError("Pilot feature user dimension does not match receiver input")
        scale = self.input_scale(received)
        received = received * scale
        if initial_channel is not None:
            initial_channel = initial_channel * scale.unsqueeze(1)
        features, pilots = self.initializer_features(
            received, pilot_features, initial_channel
        )
        return self._state_from_features(features, pilots, active_users, mcs_mask)

    def _state_from_features(
        self,
        features: Tensor,
        pilots: Tensor,
        active_users: Tensor,
        mcs_mask: Tensor | None,
    ) -> _PreparedState:
        batch, users = active_users.shape
        num_mcs = len(self.modulation_orders)
        active = active_users.to(dtype=features.dtype)
        active_rows = active.reshape(batch * users, 1, 1, 1)
        if mcs_mask is not None and mcs_mask.shape != (batch, users, num_mcs):
            raise ValueError("MCS mask must have shape [B,U,M]")
        if num_mcs == 1:
            # One modulation order: every bit is used, only inactive users are masked.
            selected_mcs = active.new_ones((batch * users, 1))
            llr_mask = active_rows
            state = self.initializer.from_features(features)
        else:
            if mcs_mask is None:
                mcs_mask = features.new_zeros((batch, users, num_mcs))
                mcs_mask[..., 0] = 1
            selected_mcs = mcs_mask.to(dtype=features.dtype).reshape(batch * users, num_mcs)
            if self.config.mcs_masking:
                state = self.initializer.from_features(features)
            else:
                initializers = (self.initializer, *self.mcs_initializers)
                state = sum(
                    initializer.from_features(features)
                    * selected_mcs[:, index, None, None, None]
                    for index, initializer in enumerate(initializers)
                )
            bit_indices = torch.arange(
                self.config.num_bits_per_symbol, device=features.device
            )
            bit_mask = torch.stack(
                [bit_indices < num_bits for num_bits in self.modulation_orders], dim=0
            )
            selected_bits = torch.matmul(selected_mcs, bit_mask.to(dtype=features.dtype))
            llr_mask = (selected_bits * active_rows[:, :, 0, 0])[:, :, None, None]
        return _PreparedState(
            state=state,
            pilots=pilots,
            mixing=_mixing_weights(active),
            selected_mcs=selected_mcs,
            llr_mask=llr_mask,
            channel_mask=active_rows,
        )
