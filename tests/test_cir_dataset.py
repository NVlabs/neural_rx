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

import torch

from neural_rx.cir_dataset import (
    CIRDataset,
    CIRDatasetChannel,
    load_cir_dataset,
    save_cir_dataset,
)


def _training_dataset() -> CIRDataset:
    coefficients = torch.zeros((8, 1, 2, 1, 1, 3, 4), dtype=torch.complex64)
    delays = torch.zeros((8, 1, 1, 3), dtype=torch.float32)
    for sample in range(8):
        coefficients[sample].fill_(complex(sample, -sample))
        delays[sample].fill_(sample)
    return CIRDataset(coefficients, delays)


def test_cir_dataset_safe_roundtrip(tmp_path: Path) -> None:
    dataset = _training_dataset()
    path = tmp_path / "channels.safetensors"

    save_cir_dataset(path, dataset)
    loaded = load_cir_dataset(path)

    torch.testing.assert_close(loaded.coefficients, dataset.coefficients)
    torch.testing.assert_close(loaded.delays, dataset.delays)
    assert loaded.coefficients.is_contiguous()
    assert loaded.delays.is_contiguous()

    prefix = load_cir_dataset(path, max_num_samples=3)
    assert prefix.num_samples == 3
    torch.testing.assert_close(prefix.coefficients, dataset.coefficients[:3])


def test_training_channel_uses_contiguous_partitions_and_shared_indices() -> None:
    channel = CIRDatasetChannel(
        _training_dataset(),
        training=True,
        num_tx=2,
        random_subsampling=False,
        generator=torch.Generator().manual_seed(7),
    )

    coefficients, delays = channel(
        batch_size=16,
        num_time_steps=4,
        sampling_frequency=15_000.0,
    )

    assert coefficients.shape == (16, 1, 2, 2, 1, 3, 4)
    assert delays.shape == (16, 1, 2, 3)
    first_user = coefficients[:, 0, 0, 0, 0, 0, 0].real
    second_user = coefficients[:, 0, 0, 1, 0, 0, 0].real
    torch.testing.assert_close(second_user - first_user, torch.full((16,), 4.0))
    assert channel.num_samples == 4


def test_training_channel_can_subsample_user_partitions_independently() -> None:
    channel = CIRDatasetChannel(
        _training_dataset(),
        training=True,
        num_tx=2,
        random_subsampling=True,
        generator=torch.Generator().manual_seed(11),
    )

    coefficients, _ = channel(32, 4, 15_000.0)

    first_user = coefficients[:, 0, 0, 0, 0, 0, 0].real
    second_user = coefficients[:, 0, 0, 1, 0, 0, 0].real
    assert torch.all((0 <= first_user) & (first_user < 4))
    assert torch.all((4 <= second_user) & (second_user < 8))
    assert torch.any(second_user - first_user != 4)


def test_evaluation_channel_preserves_aligned_trajectory_samples() -> None:
    coefficients = torch.zeros((6, 1, 1, 2, 1, 1, 2), dtype=torch.complex64)
    delays = torch.zeros((6, 1, 2, 1), dtype=torch.float32)
    for sample in range(6):
        coefficients[sample, :, :, 0].fill_(sample)
        coefficients[sample, :, :, 1].fill_(100 + sample)
        delays[sample, :, 0].fill_(sample)
        delays[sample, :, 1].fill_(100 + sample)
    dataset = CIRDataset(coefficients, delays)
    channel = CIRDatasetChannel(
        dataset,
        training=False,
        num_tx=2,
        random_subsampling=False,
        generator=torch.Generator().manual_seed(19),
    )

    sampled_coefficients, sampled_delays = channel(20, 2, 30_000.0)

    first_user = sampled_coefficients[:, 0, 0, 0, 0, 0, 0].real
    second_user = sampled_coefficients[:, 0, 0, 1, 0, 0, 0].real
    torch.testing.assert_close(second_user - first_user, torch.full((20,), 100.0))
    torch.testing.assert_close(
        sampled_delays[:, 0, 1, 0] - sampled_delays[:, 0, 0, 0],
        torch.full((20,), 100.0),
    )

