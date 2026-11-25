# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Precomputed CIR datasets and the dataset channel model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from sionna.phy.channel import ChannelModel


_DATASET_FORMAT = "neural-rx-cir"
_DATASET_VERSION = "1"


@dataclass(frozen=True)
class CIRDataset:
    """A collection of precomputed channel impulse responses.

    ``coefficients`` has shape
    ``[samples, num_rx, num_rx_ant, num_tx, num_tx_ant, paths, time]``.
    ``delays`` has shape ``[samples, num_rx, num_tx, paths]``.
    """

    coefficients: Tensor
    delays: Tensor

    def __post_init__(self) -> None:
        if self.coefficients.dtype != torch.complex64:
            raise ValueError("CIR coefficients must use torch.complex64")
        if self.delays.dtype != torch.float32:
            raise ValueError("CIR delays must use torch.float32")
        if self.coefficients.ndim != 7:
            raise ValueError("CIR coefficients must have seven dimensions")
        if self.delays.ndim != 4:
            raise ValueError("CIR delays must have four dimensions")
        expected = (
            self.coefficients.shape[0],
            self.coefficients.shape[1],
            self.coefficients.shape[3],
            self.coefficients.shape[5],
        )
        if self.delays.shape != expected:
            raise ValueError(
                "CIR delays must match the coefficient sample, receiver, transmitter, "
                "and path dimensions"
            )
        if self.coefficients.shape[0] < 1:
            raise ValueError("A CIR dataset must contain at least one sample")

    @property
    def num_samples(self) -> int:
        return self.coefficients.shape[0]

    @property
    def num_tx(self) -> int:
        return self.coefficients.shape[3]


def save_cir_dataset(path: str | Path, dataset: CIRDataset) -> None:
    """Serialize a CIR dataset with safetensors, without Python pickling."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tensors = {
        "coefficients": dataset.coefficients.detach().cpu().contiguous(),
        "delays": dataset.delays.detach().cpu().contiguous(),
    }
    metadata = {
        "format": _DATASET_FORMAT,
        "version": _DATASET_VERSION,
    }
    save_file(tensors, destination, metadata=metadata)


def load_cir_dataset(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    max_num_samples: int | None = None,
) -> CIRDataset:
    """Load a tensor-only CIR dataset from disk."""

    if max_num_samples is not None and max_num_samples < 1:
        raise ValueError("max_num_samples must be positive")
    source = Path(path)
    with safe_open(source, framework="pt", device=str(device)) as file:
        metadata = file.metadata() or {}
        if metadata.get("format") != _DATASET_FORMAT:
            raise ValueError(f"{source} is not a neural-rx CIR dataset")
        if metadata.get("version") != _DATASET_VERSION:
            raise ValueError(f"Unsupported CIR dataset version in {source}")
        if set(file.keys()) != {"coefficients", "delays"}:
            raise ValueError(f"Unexpected tensors in CIR dataset {source}")
        coefficients = file.get_tensor("coefficients")
        delays = file.get_tensor("delays")
    if max_num_samples is not None:
        coefficients = coefficients[:max_num_samples]
        delays = delays[:max_num_samples]
    return CIRDataset(coefficients=coefficients, delays=delays)


class CIRDatasetChannel(ChannelModel):
    """Sionna ``ChannelModel`` backed by a precomputed CIR dataset.

    Training data contains single-user samples and is split into contiguous,
    equally sized user partitions. Evaluation data contains all users for each
    aligned trajectory sample.
    """

    def __init__(
        self,
        dataset: CIRDataset | str | Path,
        *,
        training: bool,
        num_tx: int,
        random_subsampling: bool = True,
        max_num_samples: int | None = None,
        device: str | torch.device = "cpu",
        generator: torch.Generator | None = None,
    ) -> None:
        super().__init__(precision="single", device=str(device))
        if num_tx < 1:
            raise ValueError("num_tx must be positive")
        if isinstance(dataset, (str, Path)):
            dataset = load_cir_dataset(
                dataset,
                device=device,
                max_num_samples=max_num_samples,
            )
        elif max_num_samples is not None:
            if max_num_samples < 1:
                raise ValueError("max_num_samples must be positive")
            dataset = CIRDataset(
                coefficients=dataset.coefficients[:max_num_samples],
                delays=dataset.delays[:max_num_samples],
            )

        self._dataset = dataset
        self._training = training
        self._num_tx = num_tx
        self._random_subsampling = random_subsampling
        self._generator = generator
        self._device = torch.device(device)

        if training:
            if dataset.num_tx != 1:
                raise ValueError("Training CIR datasets must contain one transmitter per sample")
            if dataset.num_samples % num_tx != 0:
                raise ValueError("Training samples must divide evenly into user partitions")
            self._partition_size = dataset.num_samples // num_tx
        else:
            if dataset.num_tx != num_tx:
                raise ValueError("Evaluation CIR datasets must contain every configured user")
            self._partition_size = dataset.num_samples

    @property
    def num_samples(self) -> int:
        """Number of selectable samples in each partition or trajectory."""

        return self._partition_size

    def _indices(self, shape: tuple[int, ...]) -> Tensor:
        return torch.randint(
            self._partition_size,
            shape,
            device=self._device,
            generator=self._generator,
        )

    def __call__(
        self,
        batch_size: int,
        num_time_steps: int,
        sampling_frequency: float,
    ) -> tuple[Tensor, Tensor]:
        del sampling_frequency
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        stored_time_steps = self._dataset.coefficients.shape[-1]
        if num_time_steps != stored_time_steps:
            raise ValueError(
                f"Dataset has {stored_time_steps} time steps, requested {num_time_steps}"
            )

        coefficients = self._dataset.coefficients
        delays = self._dataset.delays
        if self._training:
            shared_indices = None
            if not self._random_subsampling:
                shared_indices = self._indices((batch_size,))
            user_coefficients = []
            user_delays = []
            for user in range(self._num_tx):
                indices = shared_indices
                if indices is None:
                    indices = self._indices((batch_size,))
                indices = indices + user * self._partition_size
                user_coefficients.append(coefficients[indices])
                user_delays.append(delays[indices])
            sampled_coefficients = torch.cat(user_coefficients, dim=3)
            sampled_delays = torch.cat(user_delays, dim=2)
        elif self._random_subsampling:
            user_coefficients = []
            user_delays = []
            for user in range(self._num_tx):
                indices = self._indices((batch_size,))
                user_coefficients.append(coefficients[indices, :, :, user : user + 1])
                user_delays.append(delays[indices, :, user : user + 1])
            sampled_coefficients = torch.cat(user_coefficients, dim=3)
            sampled_delays = torch.cat(user_delays, dim=2)
        else:
            indices = self._indices((batch_size,))
            sampled_coefficients = coefficients[indices]
            sampled_delays = delays[indices]
        return sampled_coefficients, sampled_delays
