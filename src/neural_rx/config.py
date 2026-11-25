# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Configuration for the PyTorch receiver core."""

from __future__ import annotations

import ast
import configparser
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReceiverConfig:
    """Neural-network settings extracted from an experiment file."""

    label: str
    num_rx_antennas: int
    num_bits_per_symbol: int
    num_iterations: int
    state_dim: int
    init_units: tuple[int, ...]
    aggregation_units: tuple[tuple[int, ...], ...]
    update_units: tuple[tuple[int, ...], ...]
    readout_units: tuple[int, ...]
    convolution: str
    initial_channel_estimate: bool
    mcs_indices: tuple[int, ...] = ()
    num_bits_per_symbol_by_mcs: tuple[int, ...] = ()
    mcs_masking: bool = False


@dataclass(frozen=True)
class TrainingPhase:
    steps: int
    learning_rate: float
    batch_size: int
    train_tx: bool
    channel_type: str
    min_ebno_db: tuple[float, ...]
    max_ebno_db: tuple[float, ...]
    channel_readout: bool
    multiloss: bool
    channel_weight: float


@dataclass(frozen=True)
class TrainingConfig:
    phases: tuple[TrainingPhase, ...]
    checkpoint_every: int
    min_ut_velocity: float
    max_ut_velocity: float
    normalize_channel: bool
    channel_type: str
    mcs_snr_db_offsets: tuple[tuple[float, ...], ...] = ()
    dataset_path: str | None = None
    random_subsampling: bool = True
    max_dataset_samples: int | None = None
    cfo_offset_ppm: float = 0.0


@dataclass(frozen=True)
class EvaluationConfig:
    min_ebno_db: float
    max_ebno_db: float
    ebno_step_db: float
    min_ut_velocity: float
    max_ut_velocity: float
    channel_type: str
    normalize_channel: bool
    n_size_bwp: int
    batch_size: int
    dataset_path: str | None = None
    random_subsampling: bool = False
    max_dataset_samples: int | None = None
    cfo_offset_ppm: float = 0.0


@dataclass(frozen=True)
class ExperimentConfig:
    receiver: ReceiverConfig
    min_num_tx: int
    max_num_tx: int
    custom_constellation: bool
    mask_pilots: bool
    randomize_pilots: bool
    rate_adjusted_snr: bool
    decoder_iterations: int
    decoder_cn_type: str
    training: TrainingConfig
    evaluation: EvaluationConfig


def _parser(path: str | Path) -> configparser.ConfigParser:
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    if not parser.read(Path(path), encoding="utf-8"):
        raise FileNotFoundError(path)
    return parser


def _get(parser: configparser.ConfigParser, section: str, option: str, default: Any = None) -> Any:
    """Evaluate a Python literal; options with a default may be omitted."""
    if default is not None and not parser.has_option(section, option):
        return default
    return ast.literal_eval(parser.get(section, option))


def _modulation_order(mcs_index: int, mcs_table: int) -> int:
    if mcs_table != 1:
        raise ValueError(f"Only PUSCH MCS table 1 is supported, got {mcs_table}")
    if 0 <= mcs_index <= 9:
        return 2
    if 10 <= mcs_index <= 16:
        return 4
    if 17 <= mcs_index <= 28:
        return 6
    raise ValueError(f"Invalid PUSCH MCS table 1 index: {mcs_index}")


def _int_tuples(value: Any) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(item) for item in row) for row in value)


def _receiver_config(parser: configparser.ConfigParser) -> ReceiverConfig:
    mcs_indices = tuple(int(value) for value in _get(parser, "system", "mcs_index"))
    mcs_table = int(_get(parser, "system", "mcs_table"))
    modulation_orders = tuple(_modulation_order(index, mcs_table) for index in mcs_indices)
    initial_chest = _get(parser, "neural_receiver", "initial_chest")
    return ReceiverConfig(
        label=_get(parser, "global", "label"),
        num_rx_antennas=int(_get(parser, "system", "num_rx_antennas")),
        num_bits_per_symbol=max(modulation_orders),
        num_iterations=int(_get(parser, "neural_receiver", "num_nrx_iter")),
        state_dim=int(_get(parser, "neural_receiver", "d_s")),
        init_units=tuple(map(int, _get(parser, "neural_receiver", "num_units_init"))),
        aggregation_units=_int_tuples(_get(parser, "neural_receiver", "num_units_agg")),
        update_units=_int_tuples(_get(parser, "neural_receiver", "num_units_state")),
        readout_units=tuple(map(int, _get(parser, "neural_receiver", "num_units_readout"))),
        convolution=_get(parser, "neural_receiver", "layer_type_conv"),
        initial_channel_estimate=initial_chest is not None and initial_chest.lower() != "none",
        mcs_indices=mcs_indices,
        num_bits_per_symbol_by_mcs=modulation_orders,
        mcs_masking=_get(parser, "neural_receiver", "mcs_var_mcs_masking", False),
    )


def load_receiver_config(path: str | Path) -> ReceiverConfig:
    """Load the network subset of an experiment configuration."""
    return _receiver_config(_parser(path))


def _dataset_options(parser: configparser.ConfigParser, section: str, suffix: str,
                     random_subsampling: bool) -> dict[str, Any]:
    samples = f"max_dataset_samples{suffix}"
    return dict(
        dataset_path=(str(_get(parser, section, f"dataset_path{suffix}"))
                      if parser.has_option(section, f"dataset_path{suffix}") else None),
        random_subsampling=_get(parser, section, f"random_subsampling{suffix}", random_subsampling),
        max_dataset_samples=(int(_get(parser, section, samples))
                             if parser.has_option(section, samples) else None),
    )


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    """Load the complete training and evaluation setup of one experiment."""

    parser = _parser(path)
    receiver = _receiver_config(parser)
    max_num_tx = int(_get(parser, "neural_receiver", "max_num_tx"))
    schedule = _get(parser, "training", "training_schedule")
    channel_type = str(_get(parser, "training", "channel_type"))
    num_phases = len(schedule["num_iter"])
    channel_types = schedule.get("channel_type", [channel_type] * num_phases)
    phases = tuple(
        TrainingPhase(
            steps=int(schedule["num_iter"][index]),
            learning_rate=float(schedule["learning_rate"][index]),
            batch_size=int(schedule["batch_size"][index]),
            train_tx=schedule["train_tx"][index],
            channel_type=str(channel_types[index]),
            min_ebno_db=tuple(map(float, schedule["min_training_snr_db"][index])),
            max_ebno_db=tuple(map(float, schedule["max_training_snr_db"][index])),
            channel_readout=schedule["double_readout"][index],
            multiloss=schedule["apply_multiloss"][index],
            channel_weight=float(schedule["weighting_double_readout"][index]),
        )
        for index in range(num_phases)
    )
    offsets = _get(parser, "training", "mcs_training_snr_db_offset",
                   [[0.0] * len(receiver.mcs_indices)] * max_num_tx)
    training = TrainingConfig(
        phases=phases,
        checkpoint_every=int(_get(parser, "training", "num_iter_train_save")),
        min_ut_velocity=float(_get(parser, "training", "min_ut_velocity")),
        max_ut_velocity=float(_get(parser, "training", "max_ut_velocity")),
        normalize_channel=_get(parser, "training", "channel_norm"),
        channel_type=channel_type,
        mcs_snr_db_offsets=tuple(tuple(map(float, row)) for row in offsets),
        cfo_offset_ppm=float(_get(parser, "training", "cfo_offset_ppm", 0.0)),
        **_dataset_options(parser, "training", "", True),
    )
    evaluation = EvaluationConfig(
        min_ebno_db=float(_get(parser, "evaluation", "snr_db_eval_min")),
        max_ebno_db=float(_get(parser, "evaluation", "snr_db_eval_max")),
        ebno_step_db=float(_get(parser, "evaluation", "snr_db_eval_stepsize")),
        min_ut_velocity=float(_get(parser, "evaluation", "min_ut_velocity_eval")),
        max_ut_velocity=float(_get(parser, "evaluation", "max_ut_velocity_eval")),
        channel_type=str(_get(parser, "evaluation", "channel_type_eval")),
        normalize_channel=_get(parser, "evaluation", "channel_norm_eval"),
        n_size_bwp=int(_get(parser, "evaluation", "n_size_bwp_eval")),
        batch_size=int(_get(parser, "evaluation", "batch_size_eval")),
        cfo_offset_ppm=float(_get(parser, "evaluation", "cfo_offset_ppm_eval", 0.0)),
        **_dataset_options(parser, "evaluation", "_eval", False),
    )
    return ExperimentConfig(
        receiver=receiver,
        min_num_tx=int(_get(parser, "neural_receiver", "min_num_tx")),
        max_num_tx=max_num_tx,
        custom_constellation=_get(parser, "neural_receiver", "custom_constellation"),
        mask_pilots=_get(parser, "neural_receiver", "mask_pilots"),
        randomize_pilots=_get(parser, "neural_receiver", "randomize_pilots"),
        rate_adjusted_snr=_get(parser, "global", "ebno"),
        decoder_iterations=int(_get(parser, "baseline", "num_bp_iter")),
        decoder_cn_type=str(_get(parser, "baseline", "cn_type")),
        training=training,
        evaluation=evaluation,
    )
