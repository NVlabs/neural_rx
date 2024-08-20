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


# Exports TensorFlow/Keras model to ONNX
# The scripts also runs a TRT latency evaluation.
# Different quantization levels can be selected towards the end of this script.

####################################################################
# Parse args
####################################################################

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-config_name", help="config filename", type=str)
parser.add_argument("-gpu", help="GPU to use", type=int, default=0)
parser.add_argument("-num_tx", help="Max number of active users", type=int, default=1)

# Parse all arguments
args = parser.parse_args()
config_name = args.config_name
gpu_num = args.gpu
num_tx = args.num_tx

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

import numpy as np

import sys
sys.path.append('../')

from utils import Parameters, NeuralReceiverONNX, load_weights

import tf2onnx
import onnx
from utils import *

##################################################################
# Load Config
##################################################################
print(f"Loading: {config_name}.")

sys_parameters = Parameters(config_name,
                            training=False,
                            system='nrx',
                            num_tx_eval=num_tx)

if len(sys_parameters.transmitters)>1:
    print("Warning: mixedMCS currently not supported." \
          "Using only first MCS scheme in list.")

# to generate test data
generator = DataGeneratorAerial(sys_parameters)
evaluator = DataEvaluator(sys_parameters)

# init neural receiver model
neural_rx = NeuralReceiverONNX(
        num_it=sys_parameters.num_nrx_iter,
        d_s=sys_parameters.d_s,
        num_units_init=sys_parameters.num_units_init,
        num_units_agg=sys_parameters.num_units_agg,
        num_units_state=sys_parameters.num_units_state,
        num_units_readout=sys_parameters.num_units_readout,
        num_bits_per_symbol=sys_parameters.transmitters[0]._num_bits_per_symbol,
        layer_type_dense = sys_parameters.layer_type_dense,
        layer_type_conv = sys_parameters.layer_type_conv,
        layer_type_readout = sys_parameters.layer_type_readout,
        nrx_dtype = sys_parameters.nrx_dtype,
        num_tx=sys_parameters.max_num_tx,
        num_rx_ant=sys_parameters.num_rx_antennas)

# define system parameters
batch_size = 1 # for real-time typically bs=1 is required
num_rx_ant = sys_parameters.num_rx_antennas
num_prbs = sys_parameters.n_size_bwp_eval
num_tx = sys_parameters.max_num_tx
num_ofdm_symbol = sys_parameters.symbol_allocation[1]

# get active DMRS positions of first PRB for each user
# ignore empty pilots if multiple CDM groups are used
rg = sys_parameters.transmitters[0].resource_grid.build_type_grid().numpy()
# value of pilots used for transmission (filter empty pilots)
pilots = sys_parameters.transmitters[0].resource_grid.pilot_pattern.pilots
pilots = pilots.numpy()

# generate NRX inputs data for dummy inference of the receiver
nrx_inputs, bits, u, h = generator(batch_size, 0.)

neural_rx(nrx_inputs);

# load weights
print("Loading pre-trained weights.")
load_weights(neural_rx, f"../weights/{sys_parameters.label}_weights")

# and set number iterations for evaluation
neural_rx._cgnn.num_it = sys_parameters.num_nrx_iter_eval

###################
# Save model via TF
###################
# Remark: this is not strictly required for the ONNX export
# Can be used for other inference pipelines

neural_rx.save(f"../onnx_models/{sys_parameters.label}_tf")

################
# Export to ONNX
################

# we re-use the previously generated dummy data to infer all input shapes
rx_slot,_,h_hat,_,dmrs_port_mask,dmrs_ofdm_pos,dmrs_subcarrier_pos = nrx_inputs

s_rx = rx_slot.shape
s_h = h_hat.shape
s_dmrs_mask = dmrs_port_mask.shape
s_dmrs_ofdm_pos = dmrs_ofdm_pos.shape
s_dmrs_subc_pos = dmrs_subcarrier_pos.shape

# activate dynamic shapes by setting shapes to None
# the dynamic ranges of these dimensions must be specified in 'trtexec' later
# in our case we target a dynamic number of subcarriers
s_rx = [1, None, num_ofdm_symbol, num_rx_ant]
s_h = [1, None, num_tx, num_rx_ant]

input_signature =[
    tf.TensorSpec(s_rx, tf.float32, name="rx_slot_real"),
    tf.TensorSpec(s_rx, tf.float32, name="rx_slot_imag"),
    tf.TensorSpec(s_h, tf.float32, name="h_hat_real"),
    tf.TensorSpec(s_h, tf.float32, name="h_hat_imag"),
    tf.TensorSpec(s_dmrs_mask, tf.float32, name="active_dmrs_ports"),
    tf.TensorSpec(s_dmrs_ofdm_pos, tf.int32, name="dmrs_ofdm_pos"),
    tf.TensorSpec(s_dmrs_subc_pos, tf.int32, name="dmrs_subcarrier_pos"),]

# convert model
print("---Converting ONNX model---")
onnx_model, _ = tf2onnx.convert.from_keras(neural_rx, input_signature)

# and save the ONNX model
print("---Saving ONNX model---")
onnx.save(onnx_model,f"../onnx_models/{sys_parameters.label}.onnx")

#################################
# compile ONNX model with TRTExec
#################################
batch_size = 1

# ONNX can support dynamic shapes.
# For this, [min, best, max] dimensions must be provided
num_prbs = [num_prbs,
            num_prbs,
            num_prbs] # number of active PRBs

num_dmrs_time = [len(dmrs_ofdm_pos[-1]),
                 len(dmrs_ofdm_pos[-1]),
                 len(dmrs_ofdm_pos[-1]),
                 ] # number of DMRS symbols in time per UE

num_dmrs_subcarrier = len(dmrs_subcarrier_pos[-1]) # number of pilots per PRB (and per ofdm symbol) per UE
num_ofdm_symbol = sys_parameters.symbol_allocation[1]

# Seems like best latency results are achieved if "opt" case equals "max"
# scenario as we mainly care about worst case latency anyhow.

num_pilots = []
for a,b in zip(num_prbs, num_dmrs_time):
    num_pilots.append(a * b * num_dmrs_subcarrier)

trt_command = f'trtexec --fp16 '\
    f'--onnx=../onnx_models/{sys_parameters.label}.onnx '\
    f'--saveEngine=../onnx_models/{sys_parameters.label}.plan '#\
    #f'--dumpProfile --separateProfileRun '
# for latest TensorRT versions, the flag "--preview=-fasterDynamicShapes0805"
# might give a few additional us latency.
# --int8 or --best is also an option for the dtype

# add shapes
for idx,s in enumerate((" --minShapes="," --optShapes="," --maxShapes=")):
    trt_command += s + \
        f'rx_slot_real:{batch_size}x{num_prbs[idx]*12}x{num_ofdm_symbol}x{num_rx_ant},'\
        f'rx_slot_imag:{batch_size}x{num_prbs[idx]*12}x{num_ofdm_symbol}x{num_rx_ant},'\
        f'h_hat_real:{batch_size}x{num_pilots[idx]}x{num_tx}x{num_rx_ant},'\
        f'h_hat_imag:{batch_size}x{num_pilots[idx]}x{num_tx}x{num_rx_ant}'
print(trt_command)
os.system(trt_command)

print("Done!")
