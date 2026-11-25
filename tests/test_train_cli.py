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

from neural_rx.cli import main
from neural_rx.config import load_receiver_config
from neural_rx.model import NeuralReceiver
from neural_rx.phy import SionnaPuschAdapter
from neural_rx.training import load_checkpoint


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_train_with_sionna_creates_research_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "nrx_rt_phy.safetensors"
    config_path = REPOSITORY_ROOT / "config" / "nrx_rt.cfg"

    result = main(
        [
            "train",
            "--config",
            str(config_path),
            "--max-steps",
            "1",
            "--batch-size",
            "1",
            "--init-checkpoint",
            str(REPOSITORY_ROOT / "weights" / "pretrained" / "nrx_rt.safetensors"),
            "--output",
            str(checkpoint),
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    metadata = load_checkpoint(NeuralReceiver(load_receiver_config(config_path)), checkpoint)
    assert metadata["steps"] == 1
    assert metadata['seed'] == 42
    assert metadata["label"] == "nrx_rt"
    assert metadata["init_checkpoint"].endswith("nrx_rt.safetensors")


def test_train_with_sionna_runs_two_phase_schedule(tmp_path: Path) -> None:
    checkpoint = tmp_path / "nrx_rt_schedule.safetensors"
    config_path = REPOSITORY_ROOT / "config" / "nrx_rt.cfg"

    result = main(
        [
            "train",
            "--config",
            str(config_path),
            "--phase-steps",
            "1",
            "1",
            "--checkpoint-every",
            "1",
            "--batch-size",
            "1",
            "--output",
            str(checkpoint),
            "--device",
            "cpu",
        ]
    )

    assert result == 0
    metadata = load_checkpoint(NeuralReceiver(load_receiver_config(config_path)), checkpoint)
    assert metadata["steps"] == 2
    assert metadata["phase"] == 2


def test_e2e_training_updates_and_checkpoints_constellation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "e2e_rt.safetensors"
    config_path = REPOSITORY_ROOT / "config" / "e2e_rt.cfg"
    initial_adapter = SionnaPuschAdapter.from_config(config_path, device="cpu")
    initial_points = initial_adapter.transmitter_parameters[0].detach().clone()

    result = main(
        [
            "train",
            "--config",
            str(config_path),
            "--phase-steps",
            "1",
            "1",
            "--batch-size",
            "1",
            "--amp",
            "--output",
            str(checkpoint),
            "--device",
            "cpu",
        ]
    )

    adapter = SionnaPuschAdapter.from_config(config_path, device="cpu")
    model = NeuralReceiver(load_receiver_config(config_path))
    metadata = load_checkpoint(
        model,
        checkpoint,
        extra_tensors=adapter.checkpoint_tensors,
    )

    assert result == 0
    assert metadata["amp"] is True
    assert metadata["steps"] == 2
    assert not torch.equal(adapter.transmitter_parameters[0], initial_points)
