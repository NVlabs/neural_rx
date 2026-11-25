# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Evaluate the neural receiver and the baselines for all users and MCSs.

Every curve is stored in ``results`` and reused by
``plot_results`` and ``plot_gp``.
"""

import argparse

import torch

from experiments.notebook_experiments import RECEIVERS, _path, run_bler
from neural_rx.config import load_experiment_config

parser = argparse.ArgumentParser()
parser.add_argument("-config_name", required=True, help="config filename")
parser.add_argument("-weights", help="checkpoint; defaults to weights/pretrained/<config>")
parser.add_argument("-num_target_block_errors", type=int, default=500)
parser.add_argument("-max_mc_iter", type=int, default=500)
parser.add_argument("-target_bler", type=float, default=0.001)
parser.add_argument("-num_cov_samples", type=int, default=100000)
parser.add_argument("-gpu", type=int, default=0)
parser.add_argument("-num_tx_eval", type=int, nargs="+",
                    help="numbers of active users; defaults to all configured")
parser.add_argument("-mcs_arr_eval_idx", type=int, default=-1,
                    help="MCS array index; -1 evaluates all MCSs")
parser.add_argument("-eval_nrx_only", action="store_true")
args = parser.parse_args()

experiment = load_experiment_config(_path(args.config_name))
num_mcs = len(experiment.receiver.mcs_indices)
users = args.num_tx_eval or range(experiment.min_num_tx, experiment.max_num_tx + 1)
mcs = range(num_mcs) if args.mcs_arr_eval_idx == -1 else [args.mcs_arr_eval_idx]
device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

for num_users in users:
    for mcs_index in mcs:
        for receiver in RECEIVERS[:1] if args.eval_nrx_only else RECEIVERS:
            run_bler(
                args.config_name, receiver=receiver, num_users=num_users,
                checkpoint_name=args.weights if receiver == "neural" else None,
                mcs_indices=[mcs_index] * num_users, max_batches=args.max_mc_iter,
                target_block_errors=args.num_target_block_errors,
                target_bler=args.target_bler, covariance_samples=args.num_cov_samples,
                device=device,
            )
