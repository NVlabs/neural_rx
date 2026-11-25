# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""BLER and BER evaluation for trained PyTorch receivers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch

from neural_rx.config import load_experiment_config
from neural_rx.model import NeuralReceiver
from neural_rx.phy import SionnaPuschAdapter
from neural_rx.training import load_checkpoint
from neural_rx.reproducibility import seed_everything


@dataclass(frozen=True)
class EvaluationResult:
    ebno_db: float
    ber: float
    bler: float
    crc_success_rate: float
    bits: int
    blocks: int
    bit_errors: int = 0
    block_errors: int = 0
    crc_successes: int = 0
    per_user: list[dict] = field(default_factory=list)


def evaluation_ebno_points(config_path: str | Path) -> list[float]:
    """Return the configured max-exclusive Eb/N0 grid."""

    evaluation = load_experiment_config(config_path).evaluation
    points = []
    point = evaluation.min_ebno_db
    while point < evaluation.max_ebno_db:
        points.append(float(point))
        point += evaluation.ebno_step_db
    return points


def evaluate_checkpoint(
    config_path: str | Path,
    checkpoint: str | Path,
    ebno_db: list[float] | None = None,
    *,
    batches: int | None = None,
    batch_size: int | None = None,
    device: str,
    target_block_errors: int | None = 500,
    target_bler: float | None = 0.001,
    early_stop: bool = True,
    mcs_indices: Sequence[int] | None = None,
    seed: int = 42,
) -> list[EvaluationResult]:
    """Evaluate one checkpoint with the configured Monte Carlo setup."""

    seed_everything(seed)
    experiment = load_experiment_config(config_path)
    model = NeuralReceiver(experiment.receiver).to(device).eval()
    adapter = SionnaPuschAdapter.from_config(
        config_path, mode="evaluation", device=device
    )
    adapter.compile_decoders()
    points = evaluation_ebno_points(config_path) if ebno_db is None else ebno_db
    num_mcs = len(experiment.receiver.num_bits_per_symbol_by_mcs)
    if mcs_indices is None:
        if num_mcs > 1:
            raise ValueError(
                "Multi-MCS evaluation requires an explicit MCS-array index "
                "for every configured user"
            )
        mcs_indices = (0,) * adapter.num_users
    if len(mcs_indices) != adapter.num_users:
        raise ValueError(
            "Evaluation MCS selection must define exactly one MCS-array index "
            f"for each of {adapter.num_users} configured users"
        )
    if any(index < 0 or index >= num_mcs for index in mcs_indices):
        raise ValueError(
            f"Evaluation MCS-array indices must be between 0 and {num_mcs - 1}"
        )
    load_checkpoint(
        model,
        checkpoint,
        device=device,
        extra_tensors=adapter.checkpoint_tensors,
    )
    selected_mcs = torch.tensor(mcs_indices, device=device)
    mcs_mask = torch.nn.functional.one_hot(
        selected_mcs, num_classes=num_mcs
    ).to(torch.float32)
    max_batches = 500 if batches is None else batches
    evaluation_batch_size = experiment.evaluation.batch_size if batch_size is None else batch_size
    results = []
    with torch.no_grad():
        for point in points:
            bit_errors = 0
            bit_count = 0
            block_errors = 0
            block_count = 0
            crc_successes = 0
            per_user = [
                dict(user=user, mcs_array_index=index, bits=0, blocks=0,
                     bit_errors=0, block_errors=0, crc_successes=0)
                for user, index in enumerate(mcs_indices)
            ]
            for _ in range(max_batches):
                batch = adapter.sample(
                    evaluation_batch_size,
                    point,
                    mcs_mask=mcs_mask[None].expand(evaluation_batch_size, -1, -1),
                    snr_mcs_index=mcs_indices[0],
                )
                llrs = model.forward_llrs(
                    batch.received,
                    batch.pilot_features,
                    batch.active_users,
                    batch.initial_channel,
                    mcs_mask=batch.mcs_mask,
                )
                information_bits = (
                    (batch.information_bits,)
                    if isinstance(batch.information_bits, torch.Tensor)
                    else batch.information_bits
                )
                for mcs_index in sorted(set(mcs_indices)):
                    decoded, crc_status = adapter.decode(llrs, mcs_index)
                    users = selected_mcs == mcs_index
                    differences = decoded[:, users] != information_bits[mcs_index][:, users]
                    bit_errors += int(differences.sum().item())
                    bit_count += differences.numel()
                    block_errors += int(differences.any(dim=-1).sum().item())
                    block_count += differences.shape[0] * differences.shape[1]
                    crc_successes += int(crc_status[:, users].sum().item())
                    for local_user, user in enumerate(torch.where(users)[0].tolist()):
                        counts = per_user[user]
                        errors = differences[:, local_user]
                        counts['bits'] += errors.numel()
                        counts['blocks'] += errors.shape[0]
                        counts['bit_errors'] += int(errors.sum().item())
                        counts['block_errors'] += int(errors.any(dim=-1).sum().item())
                        counts['crc_successes'] += int(crc_status[:, user].sum().item())
                if target_block_errors is not None and block_errors >= target_block_errors:
                    break
            for counts in per_user:
                counts['ber'] = counts['bit_errors'] / counts['bits']
                counts['bler'] = counts['block_errors'] / counts['blocks']
                counts['crc_success_rate'] = counts['crc_successes'] / counts['blocks']
            result = EvaluationResult(
                ebno_db=float(point),
                ber=bit_errors / bit_count,
                bler=block_errors / block_count,
                crc_success_rate=crc_successes / block_count,
                bits=bit_count,
                blocks=block_count,
                bit_errors=bit_errors,
                block_errors=block_errors,
                crc_successes=crc_successes,
                per_user=per_user,
            )
            results.append(result)
            if early_stop and target_bler is not None and result.bler <= target_bler:
                break
    return results
