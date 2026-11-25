# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss and persistence helpers for PyTorch receiver training."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, NamedTuple

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.nn import functional as functional

from neural_rx.config import ReceiverConfig
from neural_rx.model import NeuralReceiver, ReceiverOutput


class ReceiverLoss(NamedTuple):
    data: Tensor
    channel: Tensor
    total: Tensor


def apply_mcs_snr_offsets(
    snr_db: Tensor,
    mcs_mask: Tensor,
    offsets: Sequence[float],
    active_users: Tensor,
) -> Tensor:
    """Add the active users' MCS-specific SNR offsets."""

    offset_tensor = torch.as_tensor(offsets, dtype=snr_db.dtype, device=snr_db.device)
    per_user = torch.matmul(mcs_mask.to(snr_db), offset_tensor)
    return snr_db + (per_user * active_users.to(snr_db)).sum(dim=1)


def receiver_loss(
    outputs: Sequence[ReceiverOutput],
    bit_targets: Tensor,
    active_users: Tensor,
    *,
    data_mask: Tensor | None = None,
    bit_mask: Tensor | None = None,
    channel_target: Tensor | None = None,
    channel_weight: float = 0.0,
) -> ReceiverLoss:
    """Compute active-user data loss and optional channel-estimation loss."""

    if not outputs:
        raise ValueError("At least one receiver output is required")
    data_losses: list[Tensor] = []
    for output in outputs:
        logits = output.llrs
        if logits.shape != bit_targets.shape:
            raise ValueError("Bit targets must match the LLR shape")
        if active_users.shape != logits.shape[:2]:
            raise ValueError("Active users must match the LLR batch and user dimensions")
        eligible_grid = torch.ones_like(logits[..., 0])
        if data_mask is not None:
            if data_mask.ndim == 3:
                eligible_grid = eligible_grid * data_mask[None].to(dtype=logits.dtype)
            elif data_mask.ndim == 4:
                eligible_grid = eligible_grid * data_mask.to(dtype=logits.dtype)
            else:
                raise ValueError("Data mask must have shape [U,F,T] or [B,U,F,T]")
        selected_grid = eligible_grid * active_users.to(dtype=logits.dtype)[:, :, None, None]
        eligible_bits = eligible_grid[..., None].expand_as(logits)
        if bit_mask is not None:
            if bit_mask.ndim == 2:
                eligible_bits = eligible_bits * bit_mask[None, :, None, None].to(logits)
            elif bit_mask.ndim == 3:
                eligible_bits = eligible_bits * bit_mask[:, :, None, None].to(logits)
            else:
                raise ValueError("Bit mask must have shape [U,Q] or [B,U,Q]")
        element_loss = functional.binary_cross_entropy_with_logits(
            logits, bit_targets.to(dtype=logits.dtype), reduction="none"
        )
        bits_per_codeword = eligible_bits.sum(dim=-1).clamp_min(1.0)
        codeword_loss = (element_loss * eligible_bits).sum(dim=-1) / bits_per_codeword
        denominator = eligible_grid.sum().clamp_min(1.0)
        data_losses.append((codeword_loss * selected_grid).sum() / denominator)
    data_loss = torch.stack(data_losses).sum()

    channel_loss = data_loss.new_zeros(())
    if channel_target is not None:
        channel_losses = []
        mask = active_users.to(dtype=channel_target.dtype)[:, :, None, None]
        for output in outputs:
            if output.channel.shape != channel_target.shape:
                raise ValueError("Channel target must match channel readout shape")
            squared_error = (output.channel - channel_target).square().mean(dim=-1)
            channel_losses.append((squared_error * mask).mean())
        channel_loss = torch.stack(channel_losses).sum()

    return ReceiverLoss(
        data=data_loss,
        channel=channel_loss,
        total=data_loss + channel_weight * channel_loss,
    )


def save_checkpoint(
    model: NeuralReceiver,
    path: str | Path,
    config: ReceiverConfig,
    *,
    metadata: Mapping[str, Any] | None = None,
    extra_tensors: Mapping[str, Tensor] | None = None,
) -> None:
    """Write the weights to ``path`` and the receiver config next to it as JSON."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = dict(model.state_dict())
    for name, value in (extra_tensors or {}).items():
        tensors[f"_extra.{name}"] = value.detach().clone()
    save_file(tensors, str(path))
    payload = {**asdict(config), **(metadata or {})}
    path.with_suffix(".json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_checkpoint(
    model: NeuralReceiver,
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    extra_tensors: Mapping[str, Tensor] | None = None,
) -> dict[str, Any]:
    """Load the weights and extra tensors (e.g., a trained constellation).

    Returns the JSON metadata stored next to the checkpoint.
    """

    path = Path(path)
    tensors = load_file(str(path), device=str(device))
    extras = {
        key.removeprefix("_extra."): tensors.pop(key)
        for key in tuple(tensors)
        if key.startswith("_extra.")
    }
    if set(extras) != set(extra_tensors or {}):
        raise ValueError(
            f"Checkpoint extra tensors {sorted(extras)} do not match "
            f"the configured {sorted(extra_tensors or {})}"
        )
    model.load_state_dict(tensors)
    with torch.no_grad():
        for name, target in (extra_tensors or {}).items():
            target.copy_(extras[name].to(target))
    return json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
