#!/usr/bin/python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# evaluate BLER of NRX and baseline systems
# results are saved in files and can be visualized with the corresponding
# jupyter notebooks

####################################################################
# Parse args
####################################################################

import argparse

parser = argparse.ArgumentParser()

# the config defines the sys parameters
parser.add_argument("-config_name", help="config filename", type=str)
# limits the number of target of block errors during the simulation
parser.add_argument("-num_target_block_errors",
                    help="Number of target block errors", type=int, default=500)
parser.add_argument("-max_mc_iter",
                    help="Maximum Monte Carlo iterations",
                    type=int, default=500)
parser.add_argument("-target_bler",
                help="Early stop BLER simulations at a specific target BLER",
                type=float, default=0.001)
parser.add_argument("-num_cov_samples",
                    help="Number of samples for covariance generation", type=int, default=100000)
parser.add_argument("-gpu", help="GPU to use", type=int, default=0)
parser.add_argument("-num_tx_eval",
                    help="Number of active users",
                    type=int, nargs='+', default=-1)
parser.add_argument("-mcs_arr_eval_idx",
                    help="Select the MCS array index for evaluation. Use -1 to evaluate all MCSs.", type=int, default=-1)
parser.add_argument("-eval_nrx_only", help="Only evaluate the NN",
                    action="store_true", default=False)
parser.add_argument("-debug", help="Set debugging configuration", action="store_true", default=False)

# Parse all arguments
args = parser.parse_args()

config_name = args.config_name
max_mc_iter = args.max_mc_iter
num_target_block_errors = args.num_target_block_errors
eval_nrx_only = args.eval_nrx_only
num_cov_samples = args.num_cov_samples
gpu = args.gpu
target_bler = args.target_bler
num_tx_eval = args.num_tx_eval
mcs_arr_eval_idx = args.mcs_arr_eval_idx

distribute = None # use "all" to distribute over multiple GPUs

####################################################################
# Imports and GPU configuration
####################################################################

import os
# Avoid warnings from TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')

if distribute != "all":
    try:
        tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
        print('Only GPU number', args.gpu, 'used.')
        tf.config.experimental.set_memory_growth(gpus[args.gpu], True)
    except RuntimeError as e:
        print(e)

import sys
sys.path.append('../')

import sionna as sn
from sionna.utils import sim_ber
from utils import E2E_Model, Parameters, load_weights
import numpy as np
import pickle
from os.path import exists

if args.debug:
    tf.config.run_functions_eagerly(True)

##################################################################
# Run evaluations
##################################################################

# dummy parameters to access filename and to load results
sys_parameters = Parameters(config_name,
                            training=True,
                            system='dummy') # dummy system only to load config

# two different batch sizes can be configured
# the small one is used for the highly complex K-best-based receivers
# otherwise OOM errors occur
batch_size = sys_parameters.batch_size_eval
batch_size_small = sys_parameters.batch_size_eval_small

# results are directly saved in files
results_filename = f"{sys_parameters.label}_results"
results_filename = "../results/" + results_filename

if exists(results_filename):
    print(f"### File '{results_filename}' found. " \
          "It will be updated with the new results.")
    with open(results_filename, 'rb') as f:
        ebno_db, BERs, BLERs = pickle.load(f)
else:
    print(f"### No file '{results_filename}' found. One will be created.")
    ebno_db = np.arange(sys_parameters.snr_db_eval_min,
                        sys_parameters.snr_db_eval_max,
                        sys_parameters.snr_db_eval_stepsize)
    BERs = {}
    BLERs = {}

# evaluate for different number of active transmitters
if num_tx_eval == -1:
    num_tx_evals = np.arange(sys_parameters.min_num_tx,
                             sys_parameters.max_num_tx+1, 1)
else:
    if isinstance(num_tx_eval, int):
        num_tx_evals = [num_tx_eval]
    elif isinstance(num_tx_eval, (list, tuple)):
        num_tx_evals = num_tx_eval
    else:
        raise ValueError("num_tx_eval must be int or list of ints.")

if mcs_arr_eval_idx == -1:
    mcs_arr_eval_idxs = list(range(len(sys_parameters.mcs_index)))
else:
    if isinstance(mcs_arr_eval_idx, int):
        mcs_arr_eval_idxs = [mcs_arr_eval_idx]
    elif isinstance(mcs_arr_eval_idx, (list, tuple)):
        mcs_arr_eval_idxs = mcs_arr_eval_idx
    else:
        raise ValueError("mcs_arr_eval_idx must be int or list of ints.")

print(f"Evaluating for {num_tx_evals} active users and mcs_index elements {mcs_arr_eval_idxs}.")

# the evaluation can loop over multiple number of active DMRS ports / users
for num_tx_eval in num_tx_evals:

    # Generate covariance matrices for LMMSE-based baselines
    if not eval_nrx_only:
        print("Generating cov matrix.")
        os.system(f"python compute_cov_mat.py -config_name {config_name} -gpu {gpu} -num_samples {num_cov_samples} -num_tx_eval {num_tx_eval}")

    # Loop over all evaluation MCS indices
    for mcs_arr_eval_idx in mcs_arr_eval_idxs:

        #
        # Neural receiver
        #
        sn.Config.xla_compat = True
        sys_parameters = Parameters(config_name,
                                    training=False,
                                    num_tx_eval=num_tx_eval,
                                    system='nrx')

        # check channel types for consistency
        if sys_parameters.channel_type == 'TDL-B100':
            assert num_tx_eval == 1,\
                    "Channel model 'TDL-B100' only works with one transmitter"
        elif sys_parameters.channel_type in ("DoubleTDLlow", "DoubleTDLmedium",
                                            "DoubleTDLhigh"):
            assert num_tx_eval == 2,\
                "Channel model 'DoubleTDL' only works with two transmitters exactly"
        e2e_nn = E2E_Model(sys_parameters, training=False, mcs_arr_eval_idx=mcs_arr_eval_idx)

        print("\nRunning: " + sys_parameters.system)
        #  Run once and load the weights
        e2e_nn(1, 1.)
        filename = f'../weights/{sys_parameters.label}_weights'
        load_weights(e2e_nn, filename)

        # and set number iterations for evaluation
        e2e_nn._receiver._neural_rx.num_it = sys_parameters.num_nrx_iter_eval

        # Start sim
        ber, bler = sim_ber(e2e_nn,
                            graph_mode="xla",
                            ebno_dbs=ebno_db,
                            max_mc_iter=max_mc_iter,
                            num_target_block_errors=num_target_block_errors,
                            batch_size=batch_size,
                            distribute=distribute,
                            target_bler=target_bler,
                            early_stop=True,
                            forward_keyboard_interrupt=True)
        BERs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
        BLERs[e2e_nn._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler
        with open(results_filename, "wb") as f:
            pickle.dump([ebno_db, BERs, BLERs], f)
        sn.Config.xla_compat = False

        #
        # Baseline: LS estimation/lin interpolation + LMMSE detection
        #
        if not eval_nrx_only:
            sn.Config.xla_compat = True
            sys_parameters = Parameters(config_name,
                                        training=False,
                                        num_tx_eval=num_tx_eval,
                                        system='baseline_lslin_lmmse')
            e2e_baseline = E2E_Model(sys_parameters, training=False,
                                     mcs_arr_eval_idx=mcs_arr_eval_idx)

            print("\nRunning: " + sys_parameters.system)
            ber, bler = sim_ber(e2e_baseline,
                            graph_mode="xla",
                            ebno_dbs=ebno_db,
                            max_mc_iter=max_mc_iter,
                            num_target_block_errors=num_target_block_errors,
                            target_bler=target_bler,
                            batch_size=batch_size,
                            distribute=distribute,
                            early_stop=True,
                            forward_keyboard_interrupt=True)
            BERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
            BLERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler
            with open(results_filename, "wb") as f:
                pickle.dump([ebno_db, BERs, BLERs], f)
            sn.Config.xla_compat = False
        else:
            print("skipping LSlin & LMMSE")
        #
        # Baseline: LMMSE estimation/interpolation + K-Best detection
        #
        if not eval_nrx_only:
            sn.Config.xla_compat = False
            sys_parameters = Parameters(config_name,
                                        training=False,
                                        num_tx_eval=num_tx_eval,
                                        system = 'baseline_lmmse_kbest')
            e2e_baseline = E2E_Model(sys_parameters, training=False,
                                     mcs_arr_eval_idx=mcs_arr_eval_idx)

            print("\nRunning: " + sys_parameters.system)
            ber, bler = sim_ber(e2e_baseline,
                            graph_mode="graph",
                            ebno_dbs=ebno_db,
                            max_mc_iter=max_mc_iter,
                            num_target_block_errors=num_target_block_errors,
                            target_bler=target_bler,
                            batch_size=batch_size_small, # must be small for large PRBs
                            #distribute=distribute, # somehow does not compile
                            early_stop=True,
                            forward_keyboard_interrupt=True)
            BERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
            BLERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler
            with open(results_filename, "wb") as f:
                pickle.dump([ebno_db, BERs, BLERs], f)
            sn.Config.xla_compat = False
        else:
            print("skipping LMMSE & KBest")

        # Uncomment to simulate other baselines
        #
        # Baseline: Perfect CSI + LMMSE
        #
        # currently not evaluated
        # if not eval_nrx_only:
        #     sys_parameters = Parameters(config_name,
        #                                 training=False,
        #                                 num_tx_eval=num_tx_eval,
        #                                 system='baseline_perf_csi_lmmse')
        #     e2e_baseline = E2E_Model(sys_parameters, training=False, mcs_arr_eval_idx=mcs_arr_eval_idx)

        #     print("\nRunning: " + sys_parameters.system)
        #     ber, bler = sim_ber(e2e_baseline,
        #                     graph_mode="graph",
        #                     ebno_dbs=ebno_db,
        #                     max_mc_iter=max_mc_iter, # account for reduced bs
        #                     num_target_block_errors=num_target_block_errors,
        #                     batch_size=batch_size, # must be small due to TF bug in K-best
        #                     early_stop=True)
        #     BERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
        #     BLERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler
        #     with open(results_filename, "wb") as f:
        #         pickle.dump([ebno_db, BERs, BLERs], f)
        # else:
        #     print("skipping Perfect CSI & LMMSE")

        #
        # Baseline: LMMSE estimation/interpolation + LMMSE detection
        #
        # if not eval_nrx_only:
        #     sn.Config.xla_compat = False
        #     sys_parameters = Parameters(config_name,
        #                                 training=False,
        #                                 num_tx_eval=num_tx_eval,
        #                                 system='baseline_lmmse_lmmse')
        #     e2e_baseline = E2E_Model(sys_parameters, training=False, mcs_arr_eval_idx=mcs_arr_eval_idx)

        #     print("Running: " + sys_parameters.system)
        #     ber, bler = sim_ber(e2e_baseline,
        #                     graph_mode="graph",
        #                     ebno_dbs=ebno_db,
        #                     max_mc_iter=max_mc_iter, # account for reduced bs
        #                     num_target_block_errors=num_target_block_errors,
        #                     #target_bler=target_bler,
        #                     batch_size=batch_size_small, # must be small due to TF bug in K-best
        #                     #distribute=distribute,
        #                     early_stop=True,
        #                     forward_keyboard_interrupt=True)
        #     BERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
        #     BLERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler
        #     with open(results_filename, "wb") as f:
        #         pickle.dump([ebno_db, BERs, BLERs], f)
        #     sn.Config.xla_compat = False
        # else:
        #     print("skipping LMMSE")
        #     sys_name = f"Baseline - LMMSE+LMMSE"

        #
        # Baseline: Perfect CSI + K-Best detection
        #
        if not eval_nrx_only:
            sn.Config.xla_compat = False
            sys_parameters = Parameters(config_name,
                                        training=False,
                                        num_tx_eval=num_tx_eval,
                                        system='baseline_perf_csi_kbest')
            e2e_baseline = E2E_Model(sys_parameters, training=False,
                                     mcs_arr_eval_idx=mcs_arr_eval_idx)

            print("\nRunning: " + sys_parameters.system)
            ber, bler = sim_ber(e2e_baseline,
                            graph_mode="graph",
                            ebno_dbs=ebno_db,
                            max_mc_iter=max_mc_iter, # account for reduced bs
                            num_target_block_errors=num_target_block_errors,
                            target_bler=target_bler,
                            batch_size=batch_size_small, # must be small due to TF bug in K-best
                            distribute=distribute,
                            early_stop=True,
                            forward_keyboard_interrupt=True)
            BERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = ber
            BLERs[e2e_baseline._sys_name, num_tx_eval, mcs_arr_eval_idx] = bler
            with open(results_filename, "wb") as f:
                pickle.dump([ebno_db, BERs, BLERs], f)
            sn.Config.xla_compat = False
        else:
            print("skipping Perfect CSI & K-Best")


