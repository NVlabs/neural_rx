# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


##### Utility functions for ONNX export and training in external setups #####

from tensorflow.keras import Model
from sionna.channel import OFDMChannel, gen_single_sector_topology
from sionna.utils import BinarySource, ebnodb2no, insert_dims
from sionna.ofdm import LSChannelEstimator
from sionna.utils import flatten_dims, flatten_last_dims, compute_ber, hard_decisions, expand_to_rank
import numpy as np
import tensorflow as tf
from sionna.nr import TBDecoder


class DataGeneratorAerial(Model):
    r"""DataGeneratorAerial(sys_parameters, **kwargs)

    Generator for synthetic training data for the neural-receiver including
    labels for training in non-Sionna environments (e.g., for TRT quantization).

    Remark: shapes of the outputs are aligned with NVIDIA Aerial and not
    NVIDIA Sionna.

    Remark: This code is mostly redundant with other components of this
    repository (such as the end-to-end model). However, it mostly differs
    in the Aerial compatible shapes of the tensors.

    Parameters
    ----------
    sys_parameters : Parameters
        The system parameters.

    training: bool
        If True, pilots are randomly assigned. Defaults to False.

    Input
    -----
    batch_size : int
        Batch size of random transmit signals to be generated.

    ebno_db: float
        SNR in dB.

    num_tx: int
        Number of active transmitters (=DMRS ports)

    Output
    ------
    (nrx_inputs, bits, b, h) :
        Tuple:

    nrx_inputs: list of [rx_slot_real, rx_slot_imag, h_hat_real, h_hat_imag,
                         active_dmrs_ports, dmrs_ofdm_pos, dmrs_subcarrier_pos]
        Can be used as input for the NRX.
        The shapes are as follows:

        rx_slot_real : [bs, num_subcarrier, num_ofdm_symbols, num_rx_ant]
            Real part of channel outputs

        rx_slot_imag : [bs, num_subcarrier, num_ofdm_symbols, num_rx_ant]
            Imaginary part of channel outputs

        h_hat_real : [bs, num_pilots, num_streams, num_rx_ant]
            Real part of channel estimates at pilot positions

        h_hat_imag : [bs, num_pilots, num_streams, num_rx_ant]
            Imaginary part of channel estimates at pilot positions

        active_dmrs_ports: [batch_size, num_tx]
            Mask of 0s and 1s to indicate that DMRS ports are active or not.

        dmrs_ofdm_pos: [num_tx, num_pilot_symbols]
            Indices of DMRS pilot OFDM symbols in slot.

        dmrs_subcarrier_pos: [num,tx, num_pilots_per_prb]
            Subcarrier indices of DMRS pilots per PRB.


    c: [batch_size, num_tx, num_coded_bits], tf.float
        Encoded payload bits after TB encoding. Can be used as labels for the
        NRX training.

    b: [batch_size, num_tx, tb_size], tf.float
        Transmitted information bits on TB level (payload bits).

    h : [bs, num_rx, num_rx_ant, num_tx, num_streams_per_tx, num_ofdm_symbols,
        num_subcarrier]
            Channel frequency response (ground truth).
    """

    def __init__(self, sys_parameters, training=False):

        super().__init__()

        self._sys_parameters = sys_parameters
        self._training = training
        ###################################
        # Transmitter
        ###################################

        self._source = BinarySource()
        self._transmitter = sys_parameters.transmitters[0]

        ###################################
        # Channel
        ###################################
        self._channel = sys_parameters.channel

        ###################################
        # Receiver
        ###################################

        max_num_tx = sys_parameters.max_num_tx
        rg = sys_parameters.transmitters[0]._resource_grid

        # init channel estimator
        self._ls_est = LSChannelEstimator(resource_grid=rg,
                                          interpolation_type=None,
                                          interpolator=None)

        # precoding matrix for effective channel
        if hasattr(sys_parameters.transmitters[0], "_precoder"):
            self._w = sys_parameters.transmitters[0]._precoder._w
        else:
            self._w = tf.ones([sys_parameters.max_num_tx,
                               sys_parameters.num_antenna_ports, 1],
                               tf.complex64)

        self._w = insert_dims(self._w, 2, 1)

        # Helper to get nearest neighbor interpolation indices
        self._ls_est_nn = LSChannelEstimator(resource_grid=rg,
                                             interpolation_type="nn")

        # Precompute indices to gather received pilot signals
        self._pilots = rg.pilot_pattern.pilots
        self._pilot_ind = self._ls_est._pilot_ind
        self._ls_nn_ind = self._ls_est_nn._interpol._gather_ind

        dmrs_ofdm_pos_ = []
        dmrs_subcarrier_pos_ = []
        rg_ = sys_parameters.transmitters[0].resource_grid.build_type_grid()
        rg_np = rg_.numpy()

        for i in range(rg_np.shape[0]): # for each user individually
            idx = np.where(rg_np[i,...]==1)
            # pilot DMRS symbols (time-axis)
            dmrs_ofdm_pos_.append(np.unique(idx[1]))

            # ignore empty pilots (they are ignored in Aerial)
            p = self._pilots[i,0,:]
            # pilot DMRS subcarrier positions
            idx_active_pilots = np.where(np.abs(p)>0)[0]
            # we only focus on the first PRB (i.e. 12 subcarriers)
            idx_per_prb = idx_active_pilots[np.where(idx_active_pilots<12)]
            dmrs_subcarrier_pos_.append(idx_per_prb)

        # and stack lists for all users
        self._dmrs_ofdm_pos = tf.cast(np.stack(dmrs_ofdm_pos_), tf.int32)
        self._dmrs_subcarrier_pos = tf.cast(np.stack(dmrs_subcarrier_pos_),
                                            tf.int32)

        ###############################################
        # Pre-compute positional encoding.
        # Positional encoding consists in the distance
        # to the nearest pilot in time and frequency.
        # It is therefore a 2D positional encoding.
        ##############################################

        # Indices of the pilot-carrying resource elements and pilot symbols
        rg_type = rg.build_type_grid()[:,0] # One stream only
        pilot_ind = tf.where(rg_type==1) #
        self._pilots = rg.pilot_pattern.pilots

        # Resource grid carrying only the pilots
        # [max_num_tx, num_effective_subcarriers, num_ofdm_symbols]
        pilots_only = tf.scatter_nd(
                            pilot_ind,
                            flatten_last_dims(self._pilots , 3),
                            rg_type.shape)
        # Indices of pilots carrying RE (transmitter, freq, time)
        self.pilot_ind = tf.where(tf.abs(pilots_only) > 1e-3)

        pilot_ind = np.array(self.pilot_ind)

        # Sort the pilots according to which to which TX they are allocated
        pilot_ind_sorted = [ [] for _ in range(max_num_tx) ]

        for p_ind in pilot_ind:
            tx_ind = p_ind[0]
            re_ind = p_ind[1:]
            pilot_ind_sorted[tx_ind].append(re_ind)
        pilot_ind_sorted = np.array(pilot_ind_sorted)

        # Distance to the nearest pilot in time
        # Initialized with zeros and then filled.
        pilots_dist_time = np.zeros([   max_num_tx,
                                        rg.num_ofdm_symbols,
                                        rg.fft_size,
                                        pilot_ind_sorted.shape[1]])
        # Distance to the nearest pilot in frequency
        # Initialized with zeros and then filled
        pilots_dist_freq = np.zeros([   max_num_tx,
                                        rg.num_ofdm_symbols,
                                        rg.fft_size,
                                        pilot_ind_sorted.shape[1]])

        t_ind = np.arange(rg.num_ofdm_symbols)
        f_ind = np.arange(rg.fft_size)

        for tx_ind in range(max_num_tx):
            for i, p_ind in enumerate(pilot_ind_sorted[tx_ind]):

                pt = np.expand_dims(np.abs(p_ind[0] - t_ind), axis=1)
                pilots_dist_time[tx_ind, :, :, i] = pt

                pf = np.expand_dims(np.abs(p_ind[1] - f_ind), axis=0)
                pilots_dist_freq[tx_ind, :, :, i] = pf

        # Normalizing the tensors of distance to force zero-mean and
        # unit variance.
        nearest_pilot_dist_time = np.min(pilots_dist_time, axis=-1)
        nearest_pilot_dist_freq = np.min(pilots_dist_freq, axis=-1)
        nearest_pilot_dist_time -= np.mean(nearest_pilot_dist_time,
                                            axis=1, keepdims=True)
        std_ = np.std(nearest_pilot_dist_time, axis=1, keepdims=True)
        nearest_pilot_dist_time = np.where(std_ > 0.,
                                           nearest_pilot_dist_time / std_,
                                           nearest_pilot_dist_time)
        nearest_pilot_dist_freq -= np.mean(nearest_pilot_dist_freq,
                                            axis=2, keepdims=True)
        std_ = np.std(nearest_pilot_dist_freq, axis=2, keepdims=True)
        nearest_pilot_dist_freq = np.where(std_ > 0.,
                                           nearest_pilot_dist_freq / std_,
                                           nearest_pilot_dist_freq)

        # Stacking the time and frequency distances and casting to TF types.
        nearest_pilot_dist = np.stack([ nearest_pilot_dist_time,
                                        nearest_pilot_dist_freq],
                                        axis=-1)
        nearest_pilot_dist = tf.constant(nearest_pilot_dist, tf.float32)
        # Reshaping to match the expected shape.
        # [max_num_tx, num_subcarriers, num_ofdm_symbols, 2]
        self._nearest_pilot_dist = tf.transpose(nearest_pilot_dist,
                                                [0, 2, 1, 3])

        # Map showing the position of the nearest pilot for every user in time
        # and frequency.
        # This can be seen as a form of positional encoding
        # [num_tx, num_subcarriers, num_ofdm_symbols, 2]
        self._pe = self._nearest_pilot_dist[:self._sys_parameters.max_num_tx]

    def _active_dmrs_mask(self, batch_size, num_tx, max_num_tx):
        """Sample mask of num_tx active users
        uses different realization per batch sample"""

        max_num_tx = tf.cast(max_num_tx, tf.int32)
        num_tx = tf.cast(num_tx, tf.int32)
        r = tf.range(max_num_tx, dtype=tf.int32)
        r = tf.expand_dims(r, axis=0)
        r = tf.tile(r, (batch_size,1))
        x = tf.where(r<tf.cast(num_tx, tf.int32),
                     tf.ones_like(r),
                     tf.zeros_like(r))
        x = tf.expand_dims(x, axis=-1)
        x_p = tf.map_fn(lambda v: tf.random.shuffle(v), x)
        x_p = tf.cast(x_p, tf.float32)
        return tf.squeeze(x_p, axis=-1)

    def _set_transmitter_random_pilots(self):
        """
        Sample a random slot number and assigns its pilots to the transmitter
        """
        pilot_set = self._sys_parameters.pilots
        num_pilots = tf.shape(pilot_set)[0]
        random_pilot_ind = tf.random.uniform((), 0, num_pilots, dtype=tf.int32)
        pilots = tf.gather(pilot_set, random_pilot_ind, axis=0)
        self._transmitter.pilot_pattern.pilots = pilots

    def call(self, batch_size, ebno_db, num_tx=None):
        """Similar to end-to-end model but returns more tensors."""

        # randomly sample num_tx active users
        if num_tx is None:
            num_tx = self._sys_parameters.max_num_tx

        # generate active DMRS/user mask
        dmrs_port_mask = self._active_dmrs_mask(batch_size, num_tx,
                                                self._sys_parameters.max_num_tx)

        ###################################
        # Transmitter
        ###################################

        b = self._source([batch_size,
                          self._sys_parameters.max_num_tx,
                          self._transmitter._tb_size])

        # can be used for training and to calculate uncoded BLERs
        c = self._sys_parameters.transmitters[0]._tb_encoder(b)

        # Sample a random slot number and assigns its pilots to the transmitter
        # we randomize the pilots during training to avoid overfitting.
        if self._training:
            self._set_transmitter_random_pilots()

        x = self._transmitter(b)

        # mask non-active DMRS ports
        a_tx = expand_to_rank(dmrs_port_mask, tf.rank(x), axis=-1)
        x = tf.multiply(x, tf.cast(a_tx, tf.complex64))

        ###################################
        # Channel
        ###################################

        # Apply TX hardware impairments
        # CFO is applied per UE (i.e., must be done at TX side)
        if self._sys_parameters.frequency_offset is not None:
            x = self._sys_parameters.frequency_offset(x)

        # Rate adjusted SNR; for e2e learning non-rate adjusted is sometimes
        # preferred as pilotless communications changes the rate.
        if self._sys_parameters.ebno:
            no = ebnodb2no(
                    ebno_db,
                    self._transmitter._num_bits_per_symbol,
                    self._transmitter._target_coderate,
                    self._transmitter._resource_grid)
        else:
            no = 10**(-ebno_db/10)

        # Update topology only required for 3GPP Umi model
        if self._sys_parameters.channel_type in ("UMi", "UMa"):
            if self._sys_parameters.channel_type == "UMi":
                ch_type = 'umi'
            else:
                ch_type = 'uma'
            # Topology update only required for 3GPP pilot patterns
            topology = gen_single_sector_topology(
                        batch_size,
                        self._sys_parameters.max_num_tx,
                        ch_type,
                        min_ut_velocity=self._sys_parameters.min_ut_velocity,
                        max_ut_velocity=self._sys_parameters.max_ut_velocity,
                        indoor_probability=0.) # disable indoor users
            self._sys_parameters.channel_model.set_topology(*topology)

        y, h = self._channel([x, no])

        ####################
        # Channel estimation
        ####################

        h_hat, _ = self._ls_est((y, 0.1)) # no is arbitrary (only for err_var)
        h = tf.transpose(h, perm=[0,1,3,5,6,2,4])
        # Multiply by precoding matrices to compute effective channels
        # [s, num_rx, num_tx, num_ofdm_symbols,...
        #  ...fft_size, num_rx_ant, num_streams]
        h = tf.matmul(h, self._w)
        # Reshape
        # [bs, num_rx, num_rx_ant, num_tx, num_streams,...
        #  ...num_ofdm_symbols, fft_size]
        h = tf.transpose(h, perm=[0,1,5,2,6,3,4])

        ##################################
        # Reshape for Aerial compatibility
        ##################################

        # y has shape
        # [bs, num_rx, num_rx_ant, num_ofdm_symbols, num_subcarriers]
        # desired shape [bs, num_subcarriers, num_ofdm_symbols, num_rx_ant]
        y = tf.squeeze(y, axis=1) # remove num_rx dimension
        y = tf.transpose(y, (0,3,2,1))

        # h_hat has shape
        # [bs, num_rx, num_rx_ant, num_tx, num_streams, num_pilots]
        # desired shape is [bs, num_pilots, num_layers, num_rx_ant]
        h_hat = tf.squeeze(h_hat, axis=(1,4))

        # ignore zero pilots
        #h_hat = h_hat[...,::2] # remove 0 pilots; only works with DMRS port 0,1
        s = h_hat.shape.as_list()
        s[-1] = s[-1]//2
        h_hat = tf.gather_nd(h_hat, tf.where(tf.abs(h_hat)>1e-7))
        # recover original shape (except that every other pilot was removed
        # in last dimension)
        h_hat = tf.reshape(h_hat, s)

        h_hat = tf.transpose(h_hat, (0, 3, 2, 1))

        # no need to reshape h, as this is only required for training
        # we return numpy values, as the TRTengine is not TF-based
        nrx_inputs = [tf.math.real(y).numpy(),
                      tf.math.imag(y).numpy(),
                      tf.math.real(h_hat).numpy(),
                      tf.math.imag(h_hat).numpy(),
                      dmrs_port_mask.numpy(),
                      self._dmrs_ofdm_pos.numpy(),
                      self._dmrs_subcarrier_pos.numpy(),]
        return nrx_inputs, c, b, h


class DataEvaluator():
    """Returns loss and BER performance metric for NRX results

    Performs resource grid demapping and evaluation of receiver estimates.

    Parameters
    ----------
    sys_parameters : Parameters
        The system parameters.

    Input
    -----
    llrs : [batch_size, num_bits_per_symbol, num_tx, num_subcarriers,
            num_ofdm_symbols], tf.float
        llrs in resource grid.

    bits: [batch_size, num_tx, num_coded_bits], tf.float
        Encoded bits after TB encoding. Can be used as labels for the NRX
        training.

    Output
    ------
    (llrs, ber, u_hat) :
        Tuple:

    llrs: [batch_size, num_tx, num_coded_bits], tf.float
        One logit per codeword bit for binary classification.

    ber: tf.float
        Bit-error-rate evaluated for the given input data after a hard-decision
        of the llrs.

    u_hat: [batch_size, num_tx, num_payload_bits], tf.float
        Reconstructed information bits after TB decoding.

    """

    def __init__(self, sys_parameters):

        super().__init__()

        # store some system parameters references for easier access
        self.num_streams = sys_parameters.sm.num_streams_per_tx
        self.num_tx = sys_parameters.sm.num_tx
        self.num_bits_per_symbol \
                    = sys_parameters.transmitters[0]._num_bits_per_symbol

        rg = sys_parameters.transmitters[0]._resource_grid

        # Precompute indices to extract data symbols
        mask = rg.pilot_pattern.mask
        num_data_symbols = rg.pilot_pattern.num_data_symbols
        data_ind = tf.argsort(flatten_last_dims(mask), direction="ASCENDING")

        self.data_ind = data_ind[...,:num_data_symbols]
        self.eff_sub_ind = rg.effective_subcarrier_ind
        self.stream_ind = sys_parameters.sm.stream_ind

        self._tb_decoder = TBDecoder(sys_parameters.transmitters[0]._tb_encoder)

    def post_process_llrs(self, llr):
        # Flip the LLRs to fit the Sionna definition of LLRs
        # [batch_size, num_bits_per_symbol, num_tx, fft_size, num_ofdm_symbols]
        llr = -1.*llr

        # Undo Aerial compatibility shapes
        # Remove filler llrs (256QAM)
        # [batch_size, num_bits_per_symbol, num_tx, fft_size, num_ofdm_symbols]
        # llr = llr[:,:self.num_bits_per_symbol] # not relevant anymore

        # Extract data-carrying REs
        # Reshape to
        # [batch_size, num_tx, num_ofdm_symbols, fft_size, num_bits_per_symbol]
        llr = tf.transpose(llr, [0, 2, 4, 3, 1])

        # Remove nulled subcarriers from y (guards, dc). New shape:
        # [batch_size, num_tx, num_ofdm_symbols, num_effective_subcarriers,
        #   ... num_bits_per_symbol]
        llr = tf.gather(llr, self.eff_sub_ind, axis=-2)

        # Transpose tensor to shape
        # [num_tx, num_ofdm_symbols, num_effective_subcarriers,
        #   ... num_bits_per_symbol, batch_size]
        llr = tf.transpose(llr, [1, 2, 3, 4, 0])

        # Flatten resource grid dimensions
        # [num_tx, num_ofdm_symbols*num_effective_subcarriers,
        #  ..., num_bits_per_symbol, batch_size]
        llr = flatten_dims(llr, 2, 1)

        # Gather data symbols
        # [num_tx, num_data_symbols, num_bits_per_symbol, batch_size]
        llr = tf.gather(llr, self.data_ind[:,0,:], batch_dims=1, axis=1)

        # Put batch_dim first
        # [batch_size, num_tx, num_data_symbols, num_bits_per_symbol]
        llr = tf.transpose(llr, [3, 0, 1, 2])
        llr = llr[:,:self.num_tx]

        # Keeping only the relevant users and the unique stream per user
        # [batch_size, num_tx, num_data_symbols*num_bit_per_symbols]
        llr = flatten_last_dims(llr, 2)

        return llr

    def __call__(self, llrs, bits):

        # Post-process LLRs
        llr = self.post_process_llrs(llrs)

        # uncoded error rates
        b_hat = hard_decisions(llr)
        ber = compute_ber(b_hat, bits)

        # and apply channel decoder
        u_hat,_ = self._tb_decoder(llr)

        return llr, ber, u_hat


def precalculate_nnrx_indices(sys_parameters):
    r"""Pre-calculate static pilots and pilot indices

    This utility function precalculates the DMRS indices
    in a 5G NR PUSCH slot for the given system parameters.

    Parameters
    ----------
    sys_parameters : Parameters
        The system parameters.

    Output
    ------
    inputs: list of [pilots, pe, pilot_ind, ls_nn_ind], ndarray

        pilots: ndarray
            DMRS pilots

        pe: ndarray
            Positional encoded pilot positions

        pilot_ind: ndarray
            Indices of the DMRS pilots.

        ls_nn_ind: ndarray
            Indices for nearest neighbor interpolation in LS estimator.
    """

    ###################################
    # Receiver
    ###################################

    # from CGNNOFDMLayer
    max_num_tx = sys_parameters.max_num_tx # fixed for simplicity
    rg = sys_parameters.transmitters[0]._resource_grid

    # Integrated in the CGNN Layer (to simplify Aerial deployment)
    ls_est = LSChannelEstimator(
                    sys_parameters.transmitters[0]._resource_grid,
                    interpolation_type="nn")

    # Precompute indices to gather received pilot signals
    pilots = rg.pilot_pattern.pilots
    pilot_ind = ls_est._pilot_ind
    ls_nn_ind = ls_est._interpol._gather_ind


    ###############################################
    # Pre-compute positional encoding.
    # Positional encoding consists in the distance
    # to the nearest pilot in time and frequency.
    # It is therefore a 2D positional encoding.
    ##############################################

    # Indices of the pilot-carrying resource elements and pilot symbols
    rg_type = rg.build_type_grid()[:,0] # One stream only
    p_ind = tf.where(rg_type==1) #
    pilots = rg.pilot_pattern.pilots

    # Resource grid carrying only the pilots
    # [max_num_tx, num_effective_subcarriers, num_ofdm_symbols]
    pilots_only = tf.scatter_nd(
                        p_ind,
                        flatten_last_dims(pilots , 3),
                        rg_type.shape)
    # Indices of pilots carrying RE (transmitter, freq, time)
    p_ind = tf.where(tf.abs(pilots_only) > 1e-3)
    p_ind = np.array(p_ind)

    # Sort the pilots according to which to which TX they are allocated
    p_ind_sorted = [ [] for _ in range(max_num_tx) ]

    for p_ind in p_ind:
        tx_ind = p_ind[0]
        re_ind = p_ind[1:]
        p_ind_sorted[tx_ind].append(re_ind)
    p_ind_sorted = np.array(p_ind_sorted)

    # Distance to the nearest pilot in time
    # Initialized with zeros and then filled.
    pilots_dist_time = np.zeros([   max_num_tx,
                                    rg.num_ofdm_symbols,
                                    rg.fft_size,
                                    p_ind_sorted.shape[1]])
    # Distance to the nearest pilot in frequency
    # Initialized with zeros and then filled
    pilots_dist_freq = np.zeros([   max_num_tx,
                                    rg.num_ofdm_symbols,
                                    rg.fft_size,
                                    p_ind_sorted.shape[1]])

    t_ind = np.arange(rg.num_ofdm_symbols)
    f_ind = np.arange(rg.fft_size)

    for tx_ind in range(max_num_tx):
        for i, p_ind in enumerate(p_ind_sorted[tx_ind]):

            pt = np.expand_dims(np.abs(p_ind[0] - t_ind), axis=1)
            pilots_dist_time[tx_ind, :, :, i] = pt

            pf = np.expand_dims(np.abs(p_ind[1] - f_ind), axis=0)
            pilots_dist_freq[tx_ind, :, :, i] = pf

    # Normalizing the tensors of distance to force zero-mean and
    # unit variance.
    nearest_pilot_dist_time = np.min(pilots_dist_time, axis=-1)
    nearest_pilot_dist_freq = np.min(pilots_dist_freq, axis=-1)
    nearest_pilot_dist_time -= np.mean(nearest_pilot_dist_time,
                                        axis=1, keepdims=True)
    std_ = np.std(nearest_pilot_dist_time, axis=1, keepdims=True)
    nearest_pilot_dist_time = np.where(std_ > 0.,
                                        nearest_pilot_dist_time / std_,
                                        nearest_pilot_dist_time)
    nearest_pilot_dist_freq -= np.mean(nearest_pilot_dist_freq,
                                        axis=2, keepdims=True)
    std_ = np.std(nearest_pilot_dist_freq, axis=2, keepdims=True)
    nearest_pilot_dist_freq = np.where(std_ > 0.,
                                        nearest_pilot_dist_freq / std_,
                                        nearest_pilot_dist_freq)

    # Stacking the time and frequency distances and casting to TF types.
    nearest_pilot_dist = np.stack([ nearest_pilot_dist_time,
                                    nearest_pilot_dist_freq],
                                    axis=-1)
    nearest_pilot_dist = tf.constant(nearest_pilot_dist, tf.float32)
    # Reshaping to match the expected shape.
    # [max_num_tx, num_subcarriers, num_ofdm_symbols, 2]
    nearest_pilot_dist = tf.transpose(nearest_pilot_dist,
                                            [0, 2, 1, 3])

    # Map showing the position of the nearest pilot for every user in time
    # and frequency.
    # This can be seen as a form of positional encoding
    # [num_tx, num_subcarriers, num_ofdm_symbols, 2]
    pe = nearest_pilot_dist[:sys_parameters.max_num_tx]

    return pilots.numpy(), pe.numpy(), pilot_ind.numpy(), ls_nn_ind.numpy()
