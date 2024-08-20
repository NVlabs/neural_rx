# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Implements different channel models for performance evaluation

from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np
import sionna
from sionna.channel import GenerateOFDMChannel, ApplyOFDMChannel, ChannelModel
from sionna.channel.tr38901 import TDL

def gnb_correlation_matrix(num_ant, alpha):
    assert num_ant in [1,2,4,8]
    if num_ant==1:
        exponents = np.array([0])
    elif num_ant==2:
        exponents =  np.array([0, 1])
    elif num_ant==4:
        exponents = np.array([0, 1/9, 4/9, 1])
    elif num_ant==8:
        exponents = np.array([0, 1/49, 4/49, 9/49, 16/49, 25/49, 36/49, 1])
    row = alpha**exponents
    col = np.conj(row)
    r = tf.linalg.LinearOperatorToeplitz(col, row)
    return tf.cast(r.to_dense(), tf.complex64)

def ue_correlation_matrix(num_ant, beta):
    assert num_ant in [1,2,4]
    return gnb_correlation_matrix(num_ant, beta)

class DoubleTDLChannel(tf.keras.layers.Layer):
    """
    Channel model that stacks a 3GPP TDL-B100-400 and TDL-C-300-100 channel
    model. This allows to benchmark a two user system in a 3GPP compliant
    scenario.

    Parameters
    ---------
    carrier_frequency: float
        Carrier frequency of the simulation.

    resource_grid: ResourceGrid
        Resource grid used for the simulation.

    num_rx_ant: int
        Number of receiver antennas.

    num_tx_ant: int
        Number of transmit antennas for each user.

    norm_channel: bool
        If True, the channel is normalized.

    correlation: "low" | "medium" | "high"
        Antenna correlation according to 38.901.

    Input
    -----

    (x, no) or x:
        Tuple or Tensor:

    x :  [batch size, num_tx, num_tx_ant, num_ofdm_symbols, fft_size],
         tf.complex
        Channel inputs

    no : Scalar or Tensor, tf.float
        Scalar or tensor whose shape can be broadcast to the shape of the
        channel outputs

    Output
    -------
    y : [batch size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        Channel outputs
    h_freq : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant,
              num_ofdm_symbols, fft_size], tf.complex
        Channel frequency responses.
    """
    def __init__(self,
                 carrier_frequency,
                 resource_grid,
                 num_rx_ant=4,
                 num_tx_ant=2,
                 norm_channel=False,
                 correlation="low"):
        super().__init__()

        assert correlation in ["low", "medium", "high"]

        print(f"Loading DoubleTDL with {correlation} correlation.")

        if correlation=="low":
            alpha = beta = 0
        elif correlation=="medium":
            alpha = 0.9
            beta = 0.3
        else:
            alpha = 0.9
            beta = 0.9

        tx_corr_mat = ue_correlation_matrix(num_tx_ant, beta)
        rx_corr_mat = gnb_correlation_matrix(num_rx_ant, alpha)

        # TDL B100 model
        delay_spread_1 = 100e-9
        doppler_spread_1 = 400
        speed_1 = doppler_spread_1 * sionna.SPEED_OF_LIGHT / carrier_frequency
        tdl1 = TDL("B100",
           delay_spread_1,
           carrier_frequency,
           max_speed=speed_1,
           num_tx_ant=num_tx_ant,
           num_rx_ant=num_rx_ant,
           rx_corr_mat=rx_corr_mat,
           tx_corr_mat=tx_corr_mat)

        # TDL C300 model
        delay_spread_2 = 300e-9
        doppler_spread_2 = 100
        speed_2 = doppler_spread_2 * sionna.SPEED_OF_LIGHT / carrier_frequency
        tdl2 = TDL("C300",
           delay_spread_2,
           carrier_frequency,
           max_speed=speed_2,
           num_tx_ant=num_tx_ant,
           num_rx_ant=num_rx_ant,
           rx_corr_mat=rx_corr_mat,
           tx_corr_mat=tx_corr_mat)

        self._gen_channel_1 = GenerateOFDMChannel(
                                        tdl1,
                                        resource_grid,
                                        normalize_channel=norm_channel)

        self._gen_channel_2 = GenerateOFDMChannel(
                                        tdl2,
                                        resource_grid,
                                        normalize_channel=norm_channel)

        self._apply_channel = ApplyOFDMChannel()

    def call(self, inputs):

        x, no = inputs
        batch_size = tf.shape(x)[0]
        h1 = self._gen_channel_1(batch_size)
        h2 = self._gen_channel_2(batch_size)

        # stack the two models
        h = tf.concat([h1, h2], axis=3)

        y = self._apply_channel([x, h, no])
        return y, h

class DatasetChannel(ChannelModel):
    """Channel model from a TFRecords Dataset File
       The entire dataset is read in memory.

       This version supports XLA acceleration.


    Parameter
    ---------
    tfrecord_filename: str
        Filename of the pre-computed dataset.

    max_num_examples: int
        Max number of samples loaded from dataset. If equals to "-1"
        the entire dataset will be loaded. Defines memory occupation.

    Input
    -----
    batchsize: int
        How many samples shall be returned.

    Output
    ------
    a: [batch_size,...]
        batch_size samples from ``a``. Exact shape depends on dataset.

    tau: [batch_size,...]
        batch_size samples from ``tau``. Exact shape depends on dataset.

    """
    def __init__(self, tfrecord_filename, max_num_examples=-1, training=True,
                 num_tx=1, random_subsampling=True):

        self._training = training
        self._num_tx = num_tx
        self._random_subsampling = random_subsampling

        # Read raw dataset
        dataset = tf.data.TFRecordDataset([tfrecord_filename]) \
                  .map(self._parse_function,
                       num_parallel_calls=tf.data.AUTOTUNE) \
                  .take(max_num_examples) \
                  .batch(1024)

        # Load entire dataset into memory as large tensor
        a = None
        tau = None
        for example in dataset:
            # aggregate all channels in batch direction to multiple users.
            # i.e., move batch direction to num_tx direction.
            #
            # Evaluation data set already has two active users for each batch
            # sample.
            # Thus, every other sample after the aggregation belong to the same
            # user.
            a_ex, tau_ex = example
            a_ex = tf.split(a_ex, a_ex.shape[0], axis=0)
            a_ex = tf.concat(a_ex, axis=3)
            tau_ex = tf.split(tau_ex, tau_ex.shape[0], axis=0)
            tau_ex = tf.concat(tau_ex, axis=2)
            if a is None:
                a = a_ex
                tau = tau_ex
            else:
                a = tf.concat([a, a_ex], axis=3)
                tau = tf.concat([tau, tau_ex], axis=2)

        if training:
            # User positions are randomly sampled. In order to avoid sampling
            # the same positions multiple times within one batch sample, we
            # split the dataset into equal parts for each user to sample from
            # during simulations.
            num_examples = int(a.shape[3]/self._num_tx)
            self._num_examples = num_examples
            self._a = []
            self._tau = []
            for i in range(self._num_tx):
                self._a.append(a[:,:,:,i*num_examples:(i+1)*num_examples])
                self._tau.append(tau[:,:,i*num_examples:(i+1)*num_examples])
        else:
            self._num_examples = a.shape[3]
            self._a = [a,]
            self._tau = [tau,]

    @staticmethod
    def _parse_function(proto):
        description = {
                'a': tf.io.FixedLenFeature([], tf.string),
                'tau': tf.io.FixedLenFeature([], tf.string),
            }
        features = tf.io.parse_single_example(proto, description)
        a = tf.io.parse_tensor(features['a'], out_type=tf.complex64)
        tau = tf.io.parse_tensor(features['tau'], out_type=tf.float32)
        # tf.print(tf.shape(a))
        return a, tau


    def __call__(self, batch_size=None,
                       num_time_steps=None,
                       sampling_frequency=None):
        # default values are used for compatibility with other TF functions.

        # Remark: this is random subsampling
        # random sampling is also done in eval mode; keep in mind that even
        # though UE is on trajectory, we need many slot realizations for good
        # BLER curves (in any case we sample new AWGN noise)

        a = None
        tau = None

        if self._training:
            if not self._random_subsampling:
                ind = tf.random.uniform([batch_size],
                                     maxval=self._num_examples, dtype=tf.int32)
            # randomly subsample from different subsets
            for ue_idx in range(self._num_tx):
                if self._random_subsampling:
                    ind = tf.random.uniform(
                                        [batch_size],
                                        maxval=self._num_examples,
                                        dtype=tf.int32)

                # Gather reshape and combine
                a_ = tf.gather(self._a[ue_idx], ind, axis=3)
                a_ = tf.transpose(a_, perm=[3, 1, 2, 0, 4, 5, 6])
                tau_ = tf.gather(self._tau[ue_idx], ind, axis=2)
                tau_ = tf.transpose(tau_, perm=[2, 1, 0, 3])
                if a is not None:
                    a = tf.concat([a, a_], axis=3)
                    tau = tf.concat([tau, tau_], axis=2)
                else:
                    a = a_
                    tau = tau_
        else:
            # samples in self._a alternating between both trajectories
            if not self._random_subsampling:
                # no random sub-sampling: take subsequent two samples
                ind = tf.random.uniform([batch_size],
                                     maxval=self._num_examples//self._num_tx,
                                     dtype=tf.int32)
                ind = tf.repeat(tf.expand_dims(ind, axis=-1),
                                repeats=self._num_tx, axis=-1)
            else:
                ind = tf.random.uniform([batch_size, self._num_tx],
                                     maxval=self._num_examples//self._num_tx,
                                     dtype=tf.int32)
            # sample subsequent points from all ues
            ind = self._num_tx * ind + tf.expand_dims(
                                        tf.range(self._num_tx, dtype=tf.int32),
                                        axis=0)

            a = tf.transpose(
                    tf.squeeze(tf.gather(self._a[0], ind, axis=3), axis=0),
                    perm=[2,0,1,3,4,5,6])
            tau = tf.transpose(
                    tf.squeeze(tf.gather(self._tau[0], ind, axis=2), axis=0),
                    perm=[1,0,2,3])

        return a, tau
