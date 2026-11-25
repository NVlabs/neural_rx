# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import json
from pathlib import Path

import pytest

from neural_rx.cli import main
from neural_rx.config import load_experiment_config
from neural_rx.evaluation import evaluate_checkpoint, evaluation_ebno_points
from neural_rx.model import NeuralReceiver
from neural_rx.training import save_checkpoint


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_evaluate_checkpoint_writes_metrics(tmp_path: Path) -> None:
    source_config = REPOSITORY_ROOT / "config" / "nrx_rt.cfg"
    config_path = tmp_path / "nrx_rt.cfg"
    config_path.write_text(
        source_config.read_text(encoding="utf-8").replace(
            "n_size_bwp_eval = 273", "n_size_bwp_eval = 4"
        ),
        encoding="utf-8",
    )
    experiment = load_experiment_config(config_path)
    config = experiment.receiver
    checkpoint = tmp_path / "receiver.safetensors"
    output = tmp_path / "metrics.json"
    save_checkpoint(NeuralReceiver(config), checkpoint, config)

    result = main(
        [
            "evaluate",
            "--config",
            str(config_path),
            "--checkpoint",
            str(checkpoint),
            "--ebno-db",
            "8",
            "--batches",
            "1",
            "--batch-size",
            "1",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )

    assert result == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload['seed'] == 42
    rows = payload['results']
    assert len(rows) == 1
    assert rows[0]["ebno_db"] == 8.0
    assert 0.0 <= rows[0]["ber"] <= 1.0
    assert 0.0 <= rows[0]["bler"] <= 1.0
    assert rows[0]["blocks"] == 2
    assert sum(user['blocks'] for user in rows[0]['per_user']) == rows[0]['blocks']
    assert sum(user['bit_errors'] for user in rows[0]['per_user']) == rows[0]['bit_errors']


def test_multi_mcs_evaluation_requires_explicit_per_user_selection(
    tmp_path: Path,
) -> None:
    source_config = REPOSITORY_ROOT / "config" / "nrx_rt_var_mcs.cfg"
    config_path = tmp_path / "multi_mcs.cfg"
    config_path.write_text(
        source_config.read_text(encoding="utf-8").replace(
            "n_size_bwp_eval = 132", "n_size_bwp_eval = 4"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Multi-MCS evaluation requires an explicit"):
        evaluate_checkpoint(
            config_path,
            tmp_path / "unused.safetensors",
            [8.0],
            batches=1,
            batch_size=1,
            device="cpu",
        )


def test_evaluation_points_follow_max_exclusive_grid() -> None:
    config_path = REPOSITORY_ROOT / "config" / "nrx_rt.cfg"

    assert evaluation_ebno_points(config_path) == [
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
        7.0,
    ]


def test_mixed_mcs_counts_are_consistent_and_seeded(tmp_path):
    config_path = tmp_path / 'mixed.cfg'
    config_path.write_text((REPOSITORY_ROOT / 'config/nrx_rt_var_mcs.cfg').read_text().replace(
        'n_size_bwp_eval = 132', 'n_size_bwp_eval = 4',
    ))
    experiment = load_experiment_config(config_path)
    checkpoint = tmp_path / 'mixed.safetensors'
    save_checkpoint(NeuralReceiver(experiment.receiver), checkpoint, experiment.receiver)
    options = dict(batches=1, batch_size=2, device='cpu', mcs_indices=[0, 1], seed=17)
    first = evaluate_checkpoint(config_path, checkpoint, [8.0], **options)
    second = evaluate_checkpoint(config_path, checkpoint, [8.0], **options)
    assert first == second
    point = first[0]
    assert [user['mcs_array_index'] for user in point.per_user] == [0, 1]
    for field in ('bits', 'blocks', 'bit_errors', 'block_errors', 'crc_successes'):
        assert sum(user[field] for user in point.per_user) == getattr(point, field)
    assert point.ber == point.bit_errors / point.bits
