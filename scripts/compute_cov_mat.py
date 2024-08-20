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

# This script computes the covariance matrix for LMMSE channel estimation
# the matrices are stored in the weights/ folder

####################################################################
# Parse args
####################################################################

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-config_name", help="config filename", type=str)
parser.add_argument("-num_samples", help="Number of samples",
                    type=int, default=1000000)
parser.add_argument("-gpu", help="GPU to use", type=int, default=0)
parser.add_argument("-num_tx_eval", help="Number of active users",
                    type=int, default=1)

# Parse all arguments
args = parser.parse_args()
config_name = args.config_name
num_tx_eval = args.num_tx_eval

####################################################################
# Imports and GPU configuration
####################################################################

import os
# Avoid warnings from TensorFlow
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

import sionna as sn
sn.Config.xla_compat = True
from sionna.channel import GenerateOFDMChannel, gen_single_sector_topology

from utils import Parameters
import numpy as np

##################################################################
# Setup link
##################################################################
parameters = Parameters(config_name,
                        training=False,
                        num_tx_eval=num_tx_eval,
                        system='nrx',
                        compute_cov=True) #load UMi channel in any case

batch_size = parameters.batch_size_eval
NUM_SAMPLES = args.num_samples
# run multiple iterations to limit the batchsize/memory requirements
NUM_IT = int((NUM_SAMPLES//batch_size)+1)

channel_model = parameters.channel_model

# OFDM channel in frequency domain
gen_ofdm_channel = GenerateOFDMChannel(
                                    channel_model,
                                    parameters.transmitters[0]._resource_grid,
                                    normalize_channel=True)

#################################################################
# Evaluate covariance matrices
#################################################################

# Function that generates a batch of channel samples.
# A new topology is sampled for every batch and for every batch example.
def sample_channel(batch_size):
    # Sample a random network topology for each
    # batch example
    topology = gen_single_sector_topology(batch_size, 1, 'umi',
                                    min_ut_velocity=parameters.min_ut_velocity,
                                    max_ut_velocity=parameters.max_ut_velocity)
    channel_model.set_topology(*topology)

    # Sample channel frequency response
    # [batch size, 1, num_rx_ant, 1, 1, num_ofdm_symbols, fft_size]
    h_freq = gen_ofdm_channel(batch_size)
    # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
    h_freq = h_freq[:,0,:,0,0]

    return h_freq

@tf.function(jit_compile=True) # No XLA for better precision
def estimate_cov_mats(batch_size, num_it):
    rg = parameters.transmitters[0]._resource_grid
    freq_cov_mat = tf.zeros([rg.fft_size, rg.fft_size], tf.complex64)
    time_cov_mat = tf.zeros([rg.num_ofdm_symbols, rg.num_ofdm_symbols],
                             tf.complex64)
    space_cov_mat = tf.zeros([parameters.num_rx_antennas,
                              parameters.num_rx_antennas], tf.complex64)

    for _ in tf.range(num_it):
        # [batch size, num_rx_ant, num_ofdm_symbols, fft_size]
        h_samples = sample_channel(batch_size)
        #
        # Frequency covariance matrix estimation
        #
        # [batch size, num_rx_ant, fft_size, num_ofdm_symbols]
        h_samples_ = tf.transpose(h_samples, [0,1,3,2])
        # [batch size, num_rx_ant, fft_size, fft_size]
        freq_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [fft_size, fft_size]
        freq_cov_mat_ = tf.reduce_mean(freq_cov_mat_, axis=(0,1))
        # [fft_size, fft_size]
        freq_cov_mat += freq_cov_mat_

        #
        # Time covariance matrix estimation
        #
        # [batch size, num_rx_ant, num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.matmul(h_samples, h_samples, adjoint_b=True)
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat_ = tf.reduce_mean(time_cov_mat_, axis=(0,1))
        # [num_ofdm_symbols, num_ofdm_symbols]
        time_cov_mat += time_cov_mat_

        #
        # Spatial covariance matrix estimation
        #
        # [batch size, num_ofdm_symbols, num_rx_ant, fft_size]
        h_samples_ = tf.transpose(h_samples, [0,2,1,3])
        # [batch size, num_ofdm_symbols, num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.matmul(h_samples_, h_samples_, adjoint_b=True)
        # [num_rx_ant, num_rx_ant]
        space_cov_mat_ = tf.reduce_mean(space_cov_mat_, axis=(0,1))
        # [num_rx_ant, num_rx_ant]
        space_cov_mat += space_cov_mat_

    freq_cov_mat /= tf.complex(tf.cast(rg.num_ofdm_symbols*num_it, tf.float32),
                               0.0)
    time_cov_mat /= tf.complex(tf.cast(rg.fft_size*num_it, tf.float32), 0.0)
    space_cov_mat /= tf.complex(tf.cast(rg.fft_size*num_it, tf.float32), 0.0)
    return freq_cov_mat, time_cov_mat, space_cov_mat

#
# Run estimation
#
freq_cov_mat, time_cov_mat, space_cov_mat = estimate_cov_mats(batch_size,
                                                              NUM_IT)
freq_cov_mat = freq_cov_mat.numpy()
time_cov_mat = time_cov_mat.numpy()
space_cov_mat = space_cov_mat.numpy()

# Saving covariance matrices
# Save the time and frequency covariance matrices.
np.save(f'../weights/{parameters.label}_freq_cov_mat', freq_cov_mat)
np.save(f'../weights/{parameters.label}_time_cov_mat', time_cov_mat)
np.save(f'../weights/{parameters.label}_space_cov_mat', space_cov_mat)
