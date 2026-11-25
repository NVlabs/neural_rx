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

import pytest

from neural_rx.config import load_experiment_config, load_receiver_config


REPOSITORY_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    ("filename", "label", "iterations"),
    [
        ("nrx_rt.cfg", "nrx_rt", 2),
        ("nrx_large.cfg", "nrx_large", 8),
        ("e2e_rt.cfg", "e2e_rt", 4),
        ("e2e_large.cfg", "e2e_large", 8),
    ],
)
def test_load_receiver_config(filename: str, label: str, iterations: int) -> None:
    config = load_receiver_config(REPOSITORY_ROOT / "config" / filename)

    assert config.label == label
    assert config.num_rx_antennas == 4
    assert config.num_bits_per_symbol == 4
    assert config.num_iterations == iterations
    assert config.state_dim == (64 if filename.startswith("e2e_") else 56)
    assert len(config.aggregation_units) == iterations
    assert len(config.update_units) == iterations
    assert config.convolution == "sepconv"
    assert config.initial_channel_estimate is (not filename.startswith("e2e_"))


@pytest.mark.parametrize(
    ("filename", "mcs_indices", "modulation_orders", "masking"),
    [
        ("nrx_rt_qpsk.cfg", (9,), (2,), False),
        ("nrx_large_qpsk.cfg", (9,), (2,), False),
        ("nrx_rt_64qam.cfg", (19,), (6,), False),
        ("nrx_large_64qam.cfg", (19,), (6,), False),
        ("nrx_rt_var_mcs.cfg", (9, 14), (2, 4), False),
        ("nrx_large_var_mcs.cfg", (9, 14), (2, 4), False),
        ("nrx_large_var_mcs_64qam_masking.cfg", (9, 14, 19), (2, 4, 6), True),
    ],
)
def test_load_modulation_variants(
    filename: str,
    mcs_indices: tuple[int, ...],
    modulation_orders: tuple[int, ...],
    masking: bool,
) -> None:
    experiment = load_experiment_config(REPOSITORY_ROOT / "config" / filename)

    assert experiment.receiver.mcs_indices == mcs_indices
    assert experiment.receiver.num_bits_per_symbol_by_mcs == modulation_orders
    assert experiment.receiver.num_bits_per_symbol == max(modulation_orders)
    assert experiment.receiver.mcs_masking is masking
    assert len(experiment.training.mcs_snr_db_offsets[0]) == len(mcs_indices)


@pytest.mark.parametrize(
    ("filename", "evaluation_prbs", "multiloss"),
    [("nrx_rt.cfg", 273, False), ("nrx_large.cfg", 132, True)],
)
def test_nrx_experiment_parameters(
    filename: str, evaluation_prbs: int, multiloss: bool
) -> None:
    experiment = load_experiment_config(REPOSITORY_ROOT / "config" / filename)

    assert experiment.min_num_tx == 1
    assert experiment.max_num_tx == 2
    assert experiment.decoder_iterations == 20
    assert experiment.training.channel_type == "UMi"
    assert [phase.steps for phase in experiment.training.phases] == [
        1_000_000,
        1_000_000,
    ]
    assert [phase.channel_type for phase in experiment.training.phases] == [
        "UMi",
        "UMi",
    ]
    assert [phase.batch_size for phase in experiment.training.phases] == [128, 128]
    assert [phase.multiloss for phase in experiment.training.phases] == [
        multiloss,
        multiloss,
    ]
    assert experiment.evaluation.channel_type == "DoubleTDLlow"
    assert experiment.evaluation.n_size_bwp == evaluation_prbs
    assert experiment.evaluation.batch_size == 30


@pytest.mark.parametrize(
    ("filename", "iterations", "state_dim", "first_steps", "second_min_ebno"),
    [
        ("e2e_rt.cfg", 4, 64, 300_000, (0.0,)),
        ("e2e_large.cfg", 8, 64, 300_000, (1.0,)),
    ],
)
def test_e2e_experiment_parameters(
    filename: str,
    iterations: int,
    state_dim: int,
    first_steps: int,
    second_min_ebno: tuple[float, ...],
) -> None:
    experiment = load_experiment_config(REPOSITORY_ROOT / "config" / filename)

    assert experiment.receiver.num_iterations == iterations
    assert experiment.receiver.state_dim == state_dim
    assert experiment.min_num_tx == experiment.max_num_tx == 1
    assert experiment.custom_constellation is True
    assert experiment.mask_pilots is True
    assert experiment.training.channel_type == "TDL-C300"
    assert experiment.training.phases[0].steps == first_steps
    assert experiment.training.phases[0].channel_type == "TDL-C300"
    assert experiment.training.phases[0].train_tx is True
    assert experiment.training.phases[1].steps == 1_700_000
    assert experiment.training.phases[1].channel_type == "UMi"
    assert experiment.training.phases[1].train_tx is False
    assert experiment.training.phases[1].min_ebno_db == second_min_ebno
    assert experiment.evaluation.channel_type == "TDL-B100"
    assert experiment.evaluation.n_size_bwp == 132


def test_site_configs_expose_cir_dataset_sampling() -> None:
    fine_tuning = load_experiment_config(
        REPOSITORY_ROOT / "config" / "nrx_site_specific.cfg"
    )
    baseline = load_experiment_config(
        REPOSITORY_ROOT / "config" / "nrx_site_specific_baseline.cfg"
    )

    assert fine_tuning.training.channel_type == "Dataset"
    assert (
        fine_tuning.training.dataset_path
        == "../data/nrx_site_specific_train.safetensors"
    )
    assert fine_tuning.training.random_subsampling is False
    assert (
        fine_tuning.evaluation.dataset_path
        == "../data/nrx_site_specific_eval.safetensors"
    )
    assert fine_tuning.evaluation.random_subsampling is True
    assert baseline.training.channel_type == "UMi"
    assert [phase.steps for phase in baseline.training.phases] == [0]
    assert baseline.evaluation.channel_type == "Dataset"
