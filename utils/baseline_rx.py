# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# Implements baseline receiver algorithms for performance evaluation

from tensorflow.keras.layers import Layer
import tensorflow as tf
from itertools import combinations
import numpy as np

from sionna.ofdm import LMMSEInterpolator, KBestDetector, LinearDetector, LSChannelEstimator
from sionna.nr import PUSCHReceiver, TBDecoder, PUSCHTransmitter, PUSCHLSChannelEstimator
from sionna.utils import flatten_last_dims, split_dim, flatten_dims


class BaselineReceiver(Layer):
    """BaselineReceiver class implementing a Sionna baseline receiver for
    different receiver architectures.

    Parameters
    ----------
    sys_parameters : Parameters
        The system parameters.

    dtype : tf.complex64, optional
        The datatype of the layer, by default tf.complex64.

    return_tb_status : bool, optional
        Whether to return transport block status, by default False.

    Input
    -----
    inputs : list
        [y, no] or [y, h, no] (only for 'baseline_perf_csi')

        y : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant], tf.complex64
            The received OFDM resource grid after cyclic prefix removal and FFT.

        no : tf.float32
            Noise variance. Must have broadcastable shape to ``y``.

        h : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant], tf.complex64
            Channel frequency responses. Only required for for
            'baseline_perf_csi'.

    Output
    ------
    b_hat : [batch_size, num_tx, tb_size], tf.float32
        The reconstructed payload bits of each transport block.

    tb_crc_status : [batch_size, num_tx], tf.bool
        Transport block CRC status. Only returned if `return_tb_status`
        is `True`.
    """

    def __init__(self,
                 sys_parameters,
                 dtype=tf.complex64,
                 return_tb_status=False,
                 mcs_arr_eval_idx=0,
                 **kwargs):

        super().__init__(dtype=dtype, **kwargs)
        self._sys_parameters = sys_parameters
        self._return_tb_status = return_tb_status

        ###################################
        # Channel Estimation
        ###################################
        if sys_parameters.system in ('baseline_lmmse_kbest',
                                     'baseline_lmmse_lmmse'):
            # Setup channel estimator for non-perfect CSI

            # Use low-complexity LMMSE interpolator for large bandwidth parts
            # to keep computational complexity feasible.
            # Remark: dimensions are hard-coded in config. Needs to be adjusted
            # for different PRB dimensions.
            if sys_parameters.n_size_bwp > 100:
                print("Applying low complexity LMMSE interpolation with " \
                      "reduced number of PRBs.")

                # use automatic mode to find suitable split parameters
                if sys_parameters.lmmse_num_prbs==-1:
                    print("Using automatic LMMSE splitting.")
                    # find prime factorial of num_prbs
                    n = sys_parameters.n_size_bwp
                    prime_factors = []
                    i = 2
                    x = n
                    while i<n:
                        if x % i==0:
                            prime_factors.append(i)
                            x /= i
                        else:
                            i += 1
                    # find good split such that the number of PRBs is slightly 
                    # above 20; this is heuristic
                    n = len(prime_factors)
                    best_product = 1e6

                    for r in range(1, n+1):
                        for subset in combinations(prime_factors, r):
                            product = np.prod(subset)
                            if product > 20 and product < best_product:
                                best_product = product
                    reduction = sys_parameters.n_size_bwp / best_product
                    print(f"Using reduction factor of {reduction}")
                else:
                    reduction = sys_parameters.n_size_bwp \
                                / sys_parameters.lmmse_num_prbs

                if int(reduction)!=reduction:
                    raise ValueError("n_size_bwp must be multiple of " \
                                     "lmmse_num_prbs.")
                reduction = int(reduction)

                # modify PUSCH configs for reduced n_size_bwp
                pcs = []
                for i in range(0, len(sys_parameters.pusch_configs[mcs_arr_eval_idx])):
                    pc = sys_parameters.pusch_configs[mcs_arr_eval_idx][i].clone()
                    pc.carrier.n_size_grid = int(sys_parameters.n_size_bwp // reduction)
                    pcs.append(pc)

                self._pusch_transmitter_small = PUSCHTransmitter(pcs)
                resource_grid = self._pusch_transmitter_small.resource_grid
                pilot_pattern = resource_grid.pilot_pattern
                cov_mat_time = sys_parameters.time_cov_mat
                offset = 0
                cov_mat_freq = sys_parameters.freq_cov_mat[
                    offset:(resource_grid.fft_size + offset),
                    offset:(resource_grid.fft_size + offset)
                ]
                cov_mat_space = sys_parameters.space_cov_mat

                interpolator = LMMSEInterpolator(
                    pilot_pattern,
                    cov_mat_time=cov_mat_time,
                    cov_mat_freq=cov_mat_freq,
                    cov_mat_space=cov_mat_space,
                    order="s-f-t"
                )
                # 5G PUSCH version of low-complexity LMMSE
                self._est = LowComplexityPUSCHLMSEEstimator(
                    resource_grid=sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid,
                    dmrs_length=pc.dmrs.length,
                    dmrs_additional_position=pc.dmrs.additional_position,
                    num_cdm_groups_without_data=\
                        pc.dmrs.num_cdm_groups_without_data,
                    interpolator=interpolator,
                    sys_parameters=sys_parameters,
                    reduction=reduction
                )
            else:
                # Use standard Sionna LMMSE interpolator over all PRBs
                interpolator = LMMSEInterpolator(
                    sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid.pilot_pattern,
                    cov_mat_time=sys_parameters.time_cov_mat,
                    cov_mat_freq=sys_parameters.freq_cov_mat,
                    cov_mat_space=sys_parameters.space_cov_mat,
                    order="s-f-t"
                )
                pc = sys_parameters.pusch_configs[mcs_arr_eval_idx][0]
                self._est = PUSCHLSChannelEstimator(
                    resource_grid=sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid,
                    dmrs_length=pc.dmrs.length,
                    dmrs_additional_position=pc.dmrs.additional_position,
                    num_cdm_groups_without_data=\
                        pc.dmrs.num_cdm_groups_without_data,
                    interpolator=interpolator
                )
        elif sys_parameters.system in ('baseline_lsnn_lmmse'):
            pc = sys_parameters.pusch_configs[mcs_arr_eval_idx][0]
            self._est = PUSCHLSChannelEstimator(
                resource_grid=sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid,
                dmrs_length=pc.dmrs.length,
                dmrs_additional_position=pc.dmrs.additional_position,
                num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
                interpolation_type="nn"
            )
        elif sys_parameters.system in ('baseline_lslin_lmmse'):
            pc = sys_parameters.pusch_configs[mcs_arr_eval_idx][0]
            self._est = PUSCHLSChannelEstimator(
                resource_grid=sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid,
                dmrs_length=pc.dmrs.length,
                dmrs_additional_position=pc.dmrs.additional_position,
                num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
                interpolation_type="lin"
            )
            #self._est = LSChannelEstimator(
            #            resource_grid=sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid,
            #            interpolation_type="lin")
        elif sys_parameters.system in ('baseline_perf_csi_lmmse',
                                       'baseline_perf_csi_kbest'):
            self._est = "perfect"

        ###################################
        # Detection
        ###################################
        if sys_parameters.system in ('baseline_lmmse_kbest',
                                     'baseline_perf_csi_kbest'):
            # Init K-best detector
            self._detector = KBestDetector(
                "bit",
                sys_parameters.max_num_tx,
                64,
                sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid,
                sys_parameters.sm,
                constellation_type="qam",
                num_bits_per_symbol=\
                    sys_parameters.transmitters[mcs_arr_eval_idx]._num_bits_per_symbol
            )
        elif sys_parameters.system in ('baseline_lmmse_lmmse',
                                       'baseline_lsnn_lmmse',
                                       'baseline_lslin_lmmse',
                                       'baseline_perf_csi_lmmse'):
            # Init LMMSE detector
            self._detector = LinearDetector(
                "lmmse",
                "bit",
                sys_parameters.demapping_type,
                sys_parameters.transmitters[mcs_arr_eval_idx]._resource_grid,
                sys_parameters.sm,
                constellation_type="qam",
                num_bits_per_symbol=\
                    sys_parameters.transmitters[mcs_arr_eval_idx]._num_bits_per_symbol
            )

        ###################################
        # Decoding
        ###################################
        self._decoder = TBDecoder(
            sys_parameters.transmitters[mcs_arr_eval_idx]._tb_encoder,
            num_bp_iter=sys_parameters.num_bp_iter,
            cn_type=sys_parameters.cn_type
        )

        self._receiver = PUSCHReceiver(
            sys_parameters.transmitters[mcs_arr_eval_idx],
            channel_estimator=self._est,
            mimo_detector=self._detector,
            tb_decoder=self._decoder,
            stream_management=None,  # Will be derived from transmitters
            input_domain="freq",
            return_tb_crc_status=self._return_tb_status
        )

    def call(self, inputs):
        if self._sys_parameters.system in ("baseline_perf_csi_kbest",
                                           "baseline_perf_csi_lmmse"):
            y, h, no = inputs
            b_hat = self._receiver([y, h, no])
        else:
            y, no = inputs
            b_hat = self._receiver([y, no])
        return b_hat

# The following LMMSE estimator implementations are used to keep the
# complexity of the LMMSEEstimator class feasible.

class LowComplexityLMSEEstimator(LSChannelEstimator):
    """LowComplexityLMSEEstimator class for scalable LMMSE estimation for a
    large number of PRBs.

    The LMMSE estimation is only applied to a smaller number of PRBs
    instead of the entire number of PRBs. This leads to a small performance
    degradation, but keeps the computational (and memory) complexity
    significantly lower.
    Please note that these blocks are experimental e.g., the batch-size is
    hard-coded for XLA support (derived from sys_parameters); the same holds for
    the num_rx parameter and the number of streams per tx.

    Input
    -----
    inputs : list
        [y, no]

    y : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant], tf.complex64
        The received OFDM resource grid after cyclic prefix removal and FFT.

    no : tf.float32
        Noise variance. Must have broadcastable shape to ``y``.

    Output
    ------

    h_hat : [batch_size, 1, num_rx_ant, num_tx, 1, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates across the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variances across the entire resource grid
        for all transmitters and streams
    """


    def __init__(self, resource_grid, interpolator,
                 sys_parameters, reduction=4):
        super().__init__(resource_grid, interpolator=interpolator,
                         dtype=tf.complex64)
        self._reduction = reduction
        self._sys_parameters = sys_parameters

        # static shapes
        self._num_pilots = self._sys_parameters.transmitters[0]._resource_grid.num_pilot_symbols.numpy()

    def call(self, inputs):
        y, no = inputs
        y_eff = self._removed_nulled_scs(y)
        y_eff_flat = flatten_last_dims(y_eff)
        y_pilots = tf.gather(y_eff_flat, self._pilot_ind, axis=-1)
        h_hat, err_var = self.estimate_at_pilot_locations(y_pilots, no)
        err_var = tf.broadcast_to(err_var, tf.shape(h_hat))

        # Hard-coded batch size
        s = [self._sys_parameters.batch_size_eval_small]
        s.append(1)  # num_rx
        s.append(self._sys_parameters.num_rx_antennas)
        s.append(self._sys_parameters.max_num_tx)
        s.append(1)  # num_layer
        s.append(self._num_pilots)

        h_hat = tf.ensure_shape(h_hat, shape=s)
        err_var = tf.ensure_shape(err_var, shape=s)

        h_hat2 = split_dim(h_hat, [2, -1], axis=tf.rank(h_hat) - 1)
        h_hat3 = split_dim(h_hat2, [self._reduction, -1],
                           axis=tf.rank(h_hat2) - 1)
        h_hat4 = tf.transpose(h_hat3, perm=[6, 0, 1, 2, 3, 4, 5, 7])
        h_hat5 = flatten_last_dims(h_hat4, 2)
        h_hat6 = flatten_dims(h_hat5, 2, 0)

        err_var2 = split_dim(err_var, [2, -1], axis=tf.rank(err_var) - 1)
        err_var3 = split_dim(err_var2, [self._reduction, -1],
                             axis=tf.rank(err_var2) - 1)
        err_var4 = tf.transpose(err_var3, perm=[6, 0, 1, 2, 3, 4, 5, 7])
        err_var5 = flatten_last_dims(err_var4, 2)
        err_var6 = flatten_dims(err_var5, 2, 0)

        h_hat7, err_var7 = self._interpol(h_hat6, err_var6)
        err_var7 = tf.maximum(err_var7, tf.cast(0, err_var7.dtype))

        h_hat8 = split_dim(h_hat7, [self._reduction, -1], 0)
        h_hat9 = tf.transpose(h_hat8, perm=[1, 2, 3, 4, 5, 6, 0, 7])
        h_hat10 = flatten_last_dims(h_hat9, 2)

        err_var8 = split_dim(err_var7, [self._reduction, -1], 0)
        err_var9 = tf.transpose(err_var8, perm=[1, 2, 3, 4, 5, 6, 0, 7])
        err_var10 = flatten_last_dims(err_var9, 2)

        return h_hat10, err_var10

class LowComplexityPUSCHLMSEEstimator(PUSCHLSChannelEstimator):
    """LowComplexityPUSCHLMSEEstimator class for scalable LMMSE estimation for 5G PUSCH.

    The LMMSE estimation is only applied to a smaller number of PRBs
    instead of the entire number of PRBs. This leads to a small performance
    degradation, but keeps the computational (and memory) complexity
    significantly lower.
    Please note that these blocks are experimental e.g., the batch-size is
    hard-coded for XLA support (derived from sys_parameters); the same holds for
    the num_rx parameter and the number of streams per tx.

    Remark: Similar to LowComplexityLMMSEEstimator, but supports
    FOCC, i.e., non-orthogonal DMRS as done in  some 5G NR configurations.

    Input
    -----
    inputs : list
        [y, no]

    y : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant], tf.complex64
        The received OFDM resource grid after cyclic prefix removal and FFT.

    no : tf.float32
        Noise variance. Must have broadcastable shape to ``y``.

    Output
    ------

    h_hat : [batch_size, 1, num_rx_ant, num_tx, 1, num_ofdm_symbols, fft_size], tf.complex
        Channel estimates across the entire resource grid for all
        transmitters and streams

    err_var : Same shape as ``h_hat``, tf.float
        Channel estimation error variances across the entire resource grid
        for all transmitters and streams
    """

    def __init__(self, resource_grid, dmrs_length, dmrs_additional_position,
                 num_cdm_groups_without_data, interpolator, sys_parameters,
                 reduction=4):
        super().__init__(resource_grid=resource_grid,
                        dmrs_length=dmrs_length,
                        dmrs_additional_position=dmrs_additional_position,
                        num_cdm_groups_without_data=num_cdm_groups_without_data,
                        interpolator=interpolator,
                        dtype=tf.complex64)
        self._reduction = reduction
        self._sys_parameters = sys_parameters

        # static shapes
        self._num_pilots = self._sys_parameters.transmitters[0]._resource_grid.num_pilot_symbols.numpy()

    def call(self, inputs):
        y, no = inputs
        y_eff = self._removed_nulled_scs(y)
        y_eff_flat = flatten_last_dims(y_eff)
        y_pilots = tf.gather(y_eff_flat, self._pilot_ind, axis=-1)
        h_hat, err_var = self.estimate_at_pilot_locations(y_pilots, no)
        err_var = tf.broadcast_to(err_var, tf.shape(h_hat))

        # Hard-coded batch size
        s = [self._sys_parameters.batch_size_eval_small]
        s.append(1)  # num_rx
        s.append(self._sys_parameters.num_rx_antennas)
        s.append(self._sys_parameters.max_num_tx)
        s.append(1)  # num_layer
        s.append(self._num_pilots)

        h_hat = tf.ensure_shape(h_hat, shape=s)
        err_var = tf.ensure_shape(err_var, shape=s)

        h_hat2 = split_dim(h_hat, [2, -1], axis=tf.rank(h_hat) - 1)
        h_hat3 = split_dim(h_hat2, [self._reduction, -1],
                           axis=tf.rank(h_hat2) - 1)
        h_hat4 = tf.transpose(h_hat3, perm=[6, 0, 1, 2, 3, 4, 5, 7])
        h_hat5 = flatten_last_dims(h_hat4, 2)
        h_hat6 = flatten_dims(h_hat5, 2, 0)

        err_var2 = split_dim(err_var, [2, -1], axis=tf.rank(err_var) - 1)
        err_var3 = split_dim(err_var2, [self._reduction, -1],
                             axis=tf.rank(err_var2) - 1)
        err_var4 = tf.transpose(err_var3, perm=[6, 0, 1, 2, 3, 4, 5, 7])
        err_var5 = flatten_last_dims(err_var4, 2)
        err_var6 = flatten_dims(err_var5, 2, 0)

        h_hat7, err_var7 = self._interpol(h_hat6, err_var6)
        err_var7 = tf.maximum(err_var7, tf.cast(0, err_var7.dtype))

        h_hat8 = split_dim(h_hat7, [self._reduction, -1], 0)
        h_hat9 = tf.transpose(h_hat8, perm=[1, 2, 3, 4, 5, 6, 0, 7])
        h_hat10 = flatten_last_dims(h_hat9, 2)

        err_var8 = split_dim(err_var7, [self._reduction, -1], 0)
        err_var9 = tf.transpose(err_var8, perm=[1, 2, 3, 4, 5, 6, 0, 7])
        err_var10 = flatten_last_dims(err_var9, 2)

        return h_hat10, err_var10
