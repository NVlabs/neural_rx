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

# training of the neural receiver for a given configuration file
# the training loop can be found in utils.training_loop

####################################################################
# Parse args
####################################################################

import argparse
from os.path import exists

parser = argparse.ArgumentParser()
# the config defines the sys parameters
parser.add_argument("-config_name", help="config filename", type=str)
# GPU to use
parser.add_argument("-gpu", help="GPU to use", type=int, default=0)
# Easier debugging with breakpoints when running the code eagerly
parser.add_argument("-debug", help="Set debugging configuration", action="store_true", default=False)

# Parse all arguments
args = parser.parse_args()

####################################################################
# Imports and GPU configuration
####################################################################

# Avoid warnings from TensorFlow
import os
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

gpus = tf.config.list_physical_devices('GPU')
try:
    print('Only GPU number', args.gpu, 'used.')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except RuntimeError as e:
    print(e)

import sys
sys.path.append('../')

from utils import E2E_Model, training_loop, Parameters, load_weights

##################################################################
# Training parameters
##################################################################

# all relevant parameters are defined in the config_file
config_name = args.config_name

# initialize system parameters
sys_parameters = Parameters(config_name,
                            system='nrx',
                            training=True)
label = f'{sys_parameters.label}'
filename = '../weights/'+ label + '_weights'
training_logdir = '../logs' # use TensorBoard to visualize
training_seed = 42

if args.debug:
    tf.config.run_functions_eagerly(True)
    training_logdir = training_logdir + "/debug"

#################################################################
# Start training
#################################################################

sys_training = E2E_Model(sys_parameters, training=True)
sys_training(1, 1.) # run once to init weights in TensorFlow
sys_training.summary()

# load weights if the exists already
if exists(filename):
    print("\nWeights exist already - loading stored weights.")
    load_weights(sys_training, filename)

if hasattr(sys_parameters, 'mcs_training_snr_db_offset'):
    mcs_training_snr_db_offset = sys_parameters.mcs_training_snr_db_offset
else:
    mcs_training_snr_db_offset = None

if hasattr(sys_parameters, 'mcs_training_probs'):
    mcs_training_probs = sys_parameters.mcs_training_probs
else:
    mcs_training_probs = None

# run the training / weights are automatically saved
# UEs' MCSs will be drawn randomly
training_loop(sys_training,
              label=label,
              filename=filename,
              training_logdir=training_logdir,
              training_seed=training_seed,
              training_schedule=sys_parameters.training_schedule,
              eval_ebno_db_arr=sys_parameters.eval_ebno_db_arr,
              min_num_tx=sys_parameters.min_num_tx,
              max_num_tx=sys_parameters.max_num_tx,
              sys_parameters=sys_parameters,
              mcs_arr_training_idx=list(range(len(sys_parameters.mcs_index))), # train with all supported MCSs
              mcs_training_snr_db_offset=mcs_training_snr_db_offset,
              mcs_training_probs=mcs_training_probs,
              xla=sys_parameters.xla)
