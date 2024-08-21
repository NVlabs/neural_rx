# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

##### Neural Receiver #####

import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Conv2D, SeparableConv2D, Layer
from tensorflow.nn import relu
from sionna.utils import flatten_dims, split_dim, flatten_last_dims, insert_dims, expand_to_rank
from sionna.ofdm import ResourceGridDemapper
from sionna.nr import TBDecoder, LayerDemapper, PUSCHLSChannelEstimator

class StateInit(Layer):
    # pylint: disable=line-too-long
    r"""
    Network initializing the state tensor for each user.

    The network consist of len(num_units) hidden blocks, each block
    consisting of
    - A Separable conv layer (including a pointwise convolution)
    - A ReLU activation

    The last block is the output block and has the same architecture, but
    with `d_s` units and no non-linearity

    Parameters
    -----------
    d_s : int
        Size of the state vector

    num_units : list of int
        Number of kernels for the hidden layers of the MLP.

    layer_type: str | "sepconv" | "conv"
        Defines which Convolutional layers are used. Will be either
        SeparableConv2D or Conv2D.

    Input
    ------
    (y, pe, h_hat)
    Tuple:

    y : [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant], tf.float
        The received OFDM resource grid after cyclic prefix removal and FFT.

    pe : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
        Map showing the position of the nearest pilot for every user in time
        and frequency.
        This can be seen as a form of positional encoding.

    h_hat : None or [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
                     2*num_rx_ant], tf.float
        Initial channel estimate. If `None`, `h_hat` will be ignored.

    Output
    -------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], tf.float
        Initial state tensor for each user.
    """

    def __init__(   self,
                    d_s,
                    num_units,
                    layer_type="sepconv",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(**kwargs)

        # allows for the configuration of multiple layer types
        # one could add custom layers here
        if layer_type=="sepconv":
            layer = SeparableConv2D
        elif layer_type=="conv":
            layer = Conv2D
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Hidden blocks
        self._hidden_conv = []
        for n in num_units:
            conv = layer(n, (3,3), padding='same',
                         activation='relu', dtype=dtype)
            self._hidden_conv.append(conv)

        # Output block
        self._output_conv = layer(d_s, (3,3), activation=None,
                                  padding='same', dtype=dtype)

    def call(self, inputs):
        y, pe, h_hat = inputs

        # y : [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant],
        #  tf.float
        #  The received OFDM resource grid after cyclic prefix removal and FFT.

        # pe : [num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
        #     Map showing the position of the nearest pilot for every user in
        #     time and frequency.
        #     This can be seen as a form of positional encoding.

        # h_hat : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
        #          2*num_rx_ant], tf.float
        #     Channel estimate.

        batch_size = tf.shape(y)[0]
        num_tx = tf.shape(pe)[0]

        # Stack the inputs
        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, dim]

        y = tf.tile(tf.expand_dims(y, axis=1), [1, num_tx, 1, 1, 1])
        y = flatten_dims(y, 2, 0)

        pe = tf.tile(tf.expand_dims(pe, axis=0), [batch_size, 1, 1, 1, 1])
        pe = flatten_dims(pe, 2, 0)

        # ignore h_hat if no channel estimate is provided
        if h_hat is not None:
            h_hat = flatten_dims(h_hat, 2, 0)
            # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols,
            #   4*num_rx_ant + 3]
            z = tf.concat([y, pe, h_hat], axis=-1)
        else:
            z = tf.concat([y, pe], axis=-1)

        # Apply the neural network
        # Output : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s]
        layers = self._hidden_conv
        for conv in layers:
            z = conv(z)
        z = self._output_conv(z)

        # Unflatten
        s0 = split_dim(z, [batch_size, num_tx], 0)

        return s0 # Initial state of every user

class AggregateUserStates(Layer):
    # pylint: disable=line-too-long
    r"""
    For every user n, aggregate the states of all the other users n' != n.

    An MLP is applied to every state before aggregating.
    This is a MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a dense layer without non-linearity and with
    `d_s` units.

    The input `active_tx` provides a mask of active users and non-active users
    will be ignored in the aggregation.

    Parameters
    -----------
    d_s : int
        Size of the state vector

    num_units : list of int
        Number of units for the hidden layers.

    layer_type: str | "dense"
        Defines which Dense layers are used.

    Input
    ------
    (s, active_tx)
    Tuple:

    s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], tf.float
        Size of the state vector.

    active_tx: [batch_size, num_tx], tf.float
        Active user mask where each `0` indicates non-active users and `1`
        indicates an active user.

    Output
    -------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], tf.float
        For every user `n`, aggregate state of the other users, i.e.,
        sum(s, axis=1) - s[:,n,:,:,:]
    """

    def __init__(   self,
                    d_s,
                    num_units,
                    layer_type="dense",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(**kwargs)

        # allows for the configuration of multiple layer types
        # Currently only dense is supported
        if layer_type=="dense":
            layer = Dense
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        self._hidden_layers = []
        for n in num_units:
            self._hidden_layers.append(layer(n, activation='relu', dtype=dtype))
        self._output_layer = layer(d_s, activation=None, dtype=dtype)

    def call(self, inputs):
        r"""
        s, active_tx = inputs

        s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s],
            tf.float
            State tensor.
        active_tx: [batch_size, num_tx], tf.float
            Active user mask.
        """

        s, active_tx = inputs

        # Process s
        # Output : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s]
        sp = s
        for layer in self._hidden_layers:
            sp = layer(sp)
        sp = self._output_layer(sp)

        # Aggregate all states
        # [batch_size, 1, num_subcarriers, num_ofdm_symbols, d_s]
        # mask non active users
        active_tx = expand_to_rank(active_tx, tf.rank(sp), axis=-1)
        sp = tf.multiply(sp, active_tx)

        # aggregate and remove self-state
        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s]
        a = tf.reduce_sum(sp, axis=1, keepdims=True) - sp

        # scale by number of active users
        p = tf.reduce_sum(active_tx, axis=1, keepdims=True) - 1.
        p = tf.nn.relu(p) # clip negative values to ignore non-active user

        # avoid 0 for single active user
        p = tf.where(p==0., 1., tf.math.divide_no_nan(1.,p))

        # and scale states by number of aggregated users
        a = tf.multiply(a, p)

        return a

class UpdateState(Layer):
    # pylint: disable=line-too-long
    r"""
    Updates the state tensor.

    The network consist of len(num_units) hidden blocks, each block i
    consisting of
    - A Separable conv layer (including a pointwise convolution)
    - A ReLU activation

    The last block is the output block and has the same architecture, but
    with `d_s` units and no non-linearity.

    The network ends with a skip connection with the state.

    Parameters
    -----------
    d_s : int
        Size of the state vector.

    num_units : list of int
        Number of kernel for the hidden separable convolutional layers.

    layer_type: str | "sepconv" | "conv"
        Defines which Convolutional layers are used. Will be either
        SeparableConv2D or Conv2D.

    Input
    ------
    (s, a, pe)
    Tuple:

    s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], tf.float
        Size of the state vector.

    a : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], tf.float
        Aggregated states from other users.

    pe : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
        Map showing the position of the nearest pilot for every user in time
        and frequency. This can be seen as a form of positional encoding.

    Output
    -------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], tf.float
        Updated channel state vector.
    """

    def __init__(   self,
                    d_s,
                    num_units,
                    layer_type="sepconv",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(**kwargs)

        # allows for the configuration of multiple layer types
        if layer_type=="sepconv":
            layer = SeparableConv2D
        elif layer_type=="conv":
            layer = Conv2D
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        # Hidden blocks
        self._hidden_conv = []
        for n in num_units:
            conv = layer(n, (3,3), padding='same',
                         activation="relu", dtype=dtype)
            self._hidden_conv.append(conv)

        # Output block
        self._output_conv = layer(d_s, (3,3), padding='same',
                                  activation=None, dtype=dtype)

    def call(self, inputs):
        s, a, pe = inputs

        # s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s],
        #     tf.float
        #     State tensor.

        # a : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s],
        #     tf.float
        #     Aggregated states from other users.

        # pe : [num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
        #     Map showing the position of the nearest pilot for every user in
        #     time and frequency.
        #     This can be seen as a form of positional encoding.

        batch_size = tf.shape(s)[0]
        num_tx = tf.shape(s)[1]

        # Stack the inputs
        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, 2]
        pe = tf.tile(tf.expand_dims(pe, axis=0), [batch_size, 1, 1, 1, 1])
        pe = flatten_dims(pe, 2, 0)
        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, d_s]
        s = flatten_dims(s, 2, 0)
        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, d_s]
        a = flatten_dims(a, 2, 0)
        # [batch_size*num_tx, num_subcarriers, num_ofdm_symbols, 2*d_s + 2]
        z = tf.concat([a, s, pe], axis=-1)

        # Apply the neural network
        # Output : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s]]
        layers = self._hidden_conv
        for conv in layers:
            z = conv(z)
        z = self._output_conv(z)
        # Skip connection
        z = z + s
        # Unflatten
        s_new = split_dim(z, [batch_size, num_tx], 0)

        return s_new # Update tensor state for each user

class CGNNIt(Layer):
    # pylint: disable=line-too-long
    r"""
    Implements an iteration of the CGNN detector.

    Consists of two stages: State aggregation followed by state update.

    Parameters
    -----------
    d_s : int
        Size of the state vector.

    num_units_agg : list of int
        Number of kernel for the hidden dense layers of the aggregation network

    num_units_state_update : list of int
        Number of kernel for the hidden separable convolutional layers of the
        state-update network

    layer_type_dense: str | "dense"
        Layer type of Dense layers. Dense is used for state aggregation.

    layer_type_conv: str | "sepconv" | "conv"
        Layer type of convolutional layers. CNNs are used for state updates.

    dtype: tf.float32 | tf.float64
        Dtype of the layer.

    Input
    ------
    (s, pe, active_tx)
    Tuple:

    s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], tf.float
        Size of the state vector.

    pe : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
        Map showing the position of the nearest pilot for every user in time
        and frequency.
        This can be seen as a form of positional encoding.

    active_tx: [batch_size, num_tx], tf.float
        Active user mask where each `0` indicates non-active users and `1`
        indicates an active user.

    Output
    -------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], tf.float
        Updated channel state vector.
    """

    def __init__(   self,
                    d_s,
                    num_units_agg,
                    num_units_state_update,
                    layer_type_dense="dense",
                    layer_type_conv="sepconv",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(**kwargs)

        # Layer for state aggregation
        self._state_aggreg = AggregateUserStates(d_s,
                                                 num_units_agg,
                                                 layer_type_dense,
                                                 dtype=dtype)

        # State update
        self._state_update = UpdateState(d_s,
                                         num_units_state_update,
                                         layer_type_conv,
                                         dtype=dtype)

    def call(self, inputs):
        s, pe, active_tx = inputs

        # s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s],
        #     tf.float
        #     Size of the state vector.

        # pe : [num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
        #     Map showing the position of the nearest pilot for every user in
        #     time and frequency.
        #     This can be seen as a form of positional encoding.

        # active_tx: [batch_size, num_tx], tf.float
        #      Active user mask.

        # User state aggregation
        a = self._state_aggreg((s, active_tx))

        # State update
        s_new = self._state_update((s, a, pe))

        return s_new

class ReadoutLLRs(Layer):
    # pylint: disable=line-too-long
    r"""
    Network computing LLRs from the state vectors.

    This is a MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a dense layer without non-linearity and with
    `num_bits_per_symbol` units.

    Parameters
    -----------
    num_bits_per_symbol : int
        Number of bits per symbol.

    num_units : list of int
        Number of units for the hidden layers.

    layer_type: str | "dense"
        Defines which type of Dense layers are used.

    dtype: tf.float32 | tf.float64
        Dtype of the layer.

    Input
    ------
    s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], tf.float
        Data state.

    Output
    -------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
       num_bits_per_symbol], tf.float
        LLRs for each bit of each stream.
    """

    def __init__(   self,
                    num_bits_per_symbol,
                    num_units,
                    layer_type="dense",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(**kwargs)

       # allows for the configuration of multiple layer types
        if layer_type=="dense":
            layer = Dense
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        self._hidden_layers = []
        for n in num_units:
            self._hidden_layers.append(layer(n, activation='relu', dtype=dtype))

        self._output_layer = layer(num_bits_per_symbol,
                                   activation=None, dtype=dtype)

    def call(self, s):

        # s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s],
        #      tf.float
        #     State vector

        # Input of the MLP
        z = s
        # Apply MLP
        # Output : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
        #           num_bits_per_symbol]
        for layer in self._hidden_layers:
            z = layer(z)
        llr = self._output_layer(z)

        return llr # LLRs on the transmitted bits

class ReadoutChEst(Layer):
    # pylint: disable=line-too-long
    r"""
    Network computing channel estimate.

    This is a MLP with len(num_units) hidden layers with ReLU activation and
    num_units[i] units for the ith layer.
    The output layer is a dense layer without non-linearity and with
    `num_bits_per_symbol` units.

    Parameters
    -----------
    num_bits_per_symbol : int
        Number of bits per symbol.

    num_units : list of int
        Number of units for the hidden layers.

    layer_type: str | "dense"
        Defines which Dense layers are used.

    Input
    ------
    s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s], tf.float
        Data state.

    Output
    -------
    : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant],
      tf.float
        Channel estimate for each stream.
    """

    def __init__(   self,
                    num_rx_ant,
                    num_units,
                    layer_type="dense",
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(**kwargs)

       # allows for the configuration of multiple layer types
        if layer_type=="dense":
            layer = Dense
        else:
            raise NotImplementedError("Unknown layer_type selected.")

        self._hidden_layers = []
        for n in num_units:
            self._hidden_layers.append(layer(n, activation='relu', dtype=dtype))
        self._output_layer = layer(2*num_rx_ant, activation=None, dtype=dtype)

    def call(self, s):

        # s : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s],
        # tf.float
        #     State vector

        # Input of the MLP
        z = s
        # Apply MLP
        # Output : [batch_size, num_tx, num_subcarriers,
        #                   num_ofdm_symbols, 2*num_rx_ant]
        for layer in self._hidden_layers:
            z = layer(z)
        h_hat = self._output_layer(z)

        return h_hat # Channel estimate

class CGNN(Model):
    # pylint: disable=line-too-long
    r"""
    Implements the core neural receiver consisting of
    convolutional and graph layer components (CGNN).

    Parameters
    -----------
    num_bits_per_symbol : list of ints
        Number of bits per resource element. Defined as list for mixed MCS
        schemes.

    num_rx_ant : int
        Number of receive antennas

    num_it : int
        Number of iterations.

    d_s : int
        Size of the state vector.

    num_units_init : list of int
        Number of hidden units for the init network.

    num_units_agg : list of list of ints
        Number of kernel for the hidden dense layers of the aggregation network
        per iteration.

    num_units_state : list of list of ints
        Number of hidden units for the state-update network per iteration.

    num_units_readout : list of int
        Number of hidden units for the read-out network.

    layer_type_dense: str | "dense"
        Layer type of Dense layers.

    layer_type_conv: str | "sepconv" | "conv"
        Layer type of convolutional layers.

    layer_type_readout: str | "dense"
        Layer type of Dense readout layers.

    training : boolean
        Set to `True` if instantiated for training. Set to `False` otherwise.
        In non-training mode, the readout is only applied to the last iteration.

    dtype: tf.float32, tf.float64
        Dtype of the layer.

    Input
    ------
    (y, pe, h_hat, active_tx)
    Tuple:

    y : [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant], tf.float
        The received OFDM resource grid after cyclic prefix removal and FFT.

    pe : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
        Map showing the position of the nearest pilot for every user in time
        and frequency. This can be seen as a form of positional encoding.

    h_hat : `None` or [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
                       2*num_rx_ant], tf.float
        Initial channel estimate. If `None`, `h_hat` will be ignored.

    active_tx: [batch_size, num_tx], tf.float
        Active user mask where each `0` indicates non-active users and `1`
        indicates an active user.

    Output
    -------
    llrs : list of [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
                    num_bits_per_symbol], tf.float
        List of LLRs on (coded) bits. Each list entry refers to one iteration.
        If Training is False, only the last iteration is returned.

    h_hats : list of [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
                      2*num_rx_ant], tf.float
         List of refined channel estimates. Each list entry refers to one
         iteration. If Training is False, only the last iteration is returned.
    """

    def __init__(   self,
                    num_bits_per_symbol,
                    num_rx_ant,
                    num_it,
                    d_s,
                    num_units_init,
                    num_units_agg,
                    num_units_state ,
                    num_units_readout,
                    layer_type_dense,
                    layer_type_conv,
                    layer_type_readout,
                    training=False,
                    apply_multiloss=False,
                    var_mcs_masking=False,
                    dtype=tf.float32,
                    **kwargs):
        super().__init__(dtype=dtype,**kwargs)

        self._training = training

        self._apply_multiloss = apply_multiloss
        self._var_mcs_masking = var_mcs_masking

        # Initialization for the state
        if self._var_mcs_masking:
            self._s_init = [StateInit(  d_s,
                                num_units_init,
                                layer_type=layer_type_conv,
                                dtype=dtype)]
        else:
            self._s_init = []
            for _ in num_bits_per_symbol:
                self._s_init.append(
                    StateInit(  d_s,
                                num_units_init,
                                layer_type=layer_type_conv,
                                dtype=dtype))

        # Iterations blocks
        self._iterations = []
        for i in range(num_it):
            it = CGNNIt(    d_s,
                            num_units_agg[i],
                            num_units_state[i],
                            layer_type_dense=layer_type_dense,
                            layer_type_conv=layer_type_conv,
                            dtype=dtype)
            self._iterations.append(it)
        self._num_it = num_it

        # Readouts
        if self._var_mcs_masking:
            self._readout_llrs = [ReadoutLLRs(np.max(num_bits_per_symbol),
                                            num_units_readout,
                                            layer_type=layer_type_readout,
                                            dtype=dtype)]
        else:
            self._readout_llrs = []
            for num_bits in num_bits_per_symbol:
                self._readout_llrs.append(
                    ReadoutLLRs(num_bits,
                                            num_units_readout,
                                            layer_type=layer_type_readout,
                                            dtype=dtype))
        # The h_hat readout is mostly used for faster training convergence
        # by using a second loss on the channel estimate.
        # However, it can be also used after deployment, e.g., for reporting to
        # higher layers.
        self._readout_chest = ReadoutChEst(num_rx_ant,
                                           num_units_readout,
                                           layer_type=layer_type_readout,
                                           dtype=dtype)

        self._num_mcss_supported = len(num_bits_per_symbol)
        self._num_bits_per_symbol = num_bits_per_symbol

    @property
    def apply_multiloss(self):
        """Average loss over all iterations or eval just the last iteration."""
        return self._apply_multiloss

    @apply_multiloss.setter
    def apply_multiloss(self, val):
        assert isinstance(val, bool), "apply_multiloss must be bool."
        self._apply_multiloss = val

    @property
    def num_it(self):
        """Number of receiver iterations."""
        return self._num_it

    @num_it.setter
    def num_it(self, val):
        assert (val >= 1) and (val <= len(self._iterations)),\
            "Invalid number of iterations"
        self._num_it = val

    def call(self, inputs):
        y, pe, h_hat, active_tx, mcs_ue_mask = inputs

        # y : [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant],
        #     tf.float
        #   The received OFDM resource grid after cyclic prefix removal and FFT.

        # pe : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2],
        #      tf.float
        #  Map showing the position of the nearest pilot for every user in time
        #     and frequency.
        #     This can be seen as a form of positional encoding.

        # h_hat : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
        #          2*num_rx_ant], tf.float
        #     Channel estimate.

        # active_tx: [batch_size, num_tx], tf.float
        #      Active user mask.

        ########################################
        # Normalization
        #########################################
        # we normalize the input such that each batch sample has unit power
        # [batch_size, 1, 1, 1]
        norm_scaling = tf.reduce_mean(tf.square(y), axis=(1,2,3), keepdims=True)
        norm_scaling = tf.math.divide_no_nan(1., tf.sqrt(norm_scaling))
        # [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        y = y*norm_scaling
        # [batch_size, 1, 1, 1, 1]
        norm_scaling = tf.expand_dims(norm_scaling, axis=1)
        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        if h_hat is not None:
            h_hat = h_hat*norm_scaling

        ########################################
        # State initialization
        ########################################

        # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s]
        if self._var_mcs_masking:
            s = self._s_init[0]((y, pe, h_hat))
        else:
            s = self._s_init[0]((y, pe, h_hat)) * expand_to_rank(
                        tf.gather(mcs_ue_mask, indices=0, axis=2), 5, axis=-1)
            for idx in range(1, self._num_mcss_supported):
                s = s + self._s_init[idx]((y, pe, h_hat)) * expand_to_rank(
                        tf.gather(mcs_ue_mask, indices=idx, axis=2), 5, axis=-1)

        ########################################
        # Run receiver iterations
        ########################################
        # Remark: each iteration uses a different NN with different weights
        # weight sharing could possibly be used, but degrades the performance
        llrs = []
        h_hats = []
        for i in range(self._num_it):
            it = self._iterations[i]
            # State update
            # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, d_s]
            s = it([s, pe, active_tx])

            # Read-outs
            # [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
            #  num_bits_per_symbol]
            # only during training every intermediate iteration is tracked
            if (self._training and self._apply_multiloss) or i==self._num_it-1:
                llrs_ = []
                # iterate over all MCS schemes individually
                for idx in range(self._num_mcss_supported):
                    if self._var_mcs_masking:
                        llrs__ = self._readout_llrs[0](s)
                        # Masking of LLR outputs to the desired
                        # num_bits_per_symbol[idx] (for all users and all MCS
                        # schemes)
                        llrs__ = tf.gather(
                            llrs__,
                            indices=tf.range(self._num_bits_per_symbol[idx]),
                            axis=-1)
                    else:
                        llrs__ = self._readout_llrs[idx](s)
                    llrs_.append(llrs__)
                llrs.append(llrs_)
                h_hats.append(self._readout_chest(s))

        return llrs, h_hats

class CGNNOFDM(Model):
    # pylint: disable=line-too-long
    r"""
    Wrapper for the neural receiver (CGNN) layer that handles
    OFDM waveforms and the resourcegrid mapping/demapping.

    Layer also integrates loss function computation.

    Parameters
    -----------
    sys_parameters : Parameters
        The system parameters.

    max_num_tx : int
        Maximum number of transmitters

    training : boolean
        Set to `True` if instantiated for training. Set to `False` otherwise.

    num_it : int
        Number of iterations.

    d_s : int
        Size of the state vector.

    num_units_init : list of int
        Number of hidden units for the init network.

    num_units_agg : list of int
        Number of kernel for the hidden dense layers of the aggregation network.

    num_units_state : list of int
        Number of hidden units for the state-update network.

    num_units_readout : list of int
        Number of hidden units for the read-out network.

    layer_type_dense: str | "dense"
        Layer type of Dense layers.

    layer_type_conv: str | "sepconv" | "conv"
        Layer type of convolutional layers.

    layer_type_readout: str | "dense"
        Layer type of Dense readout layers.

    nrx_dtype: DType
        DType of the NRX layers.

    Input
    ------
    (y, h_hat, active_tx, [bits, h, mcs_ue_mask]) :
        Tuple:
    y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
        The received OFDM resource grid after cyclic prefix removal and FFT.

    h_hat : `None` or [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
                       2*num_rx_ant], tf.float
        Initial channel estimate. If `None`, `h_hat` will be ignored.

    active_tx: [batch_size, num_tx], tf.float
        Active user mask where each `0` indicates non-active users and `1`
        indicates an active user.

    bits : [batch_size, num_tx, num_data_symbols*num_bits_per_symbol], tf.int
        Transmitted bits.
        Only required for training to compute the loss function.

    h : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant],
        tf.float
        Ground-truth channel impulse response.
        Only required for training to compute the loss function.

    mcs_ue_mask: [batch_size, max_num_tx, len(mcs_index)], tf.int32
        One-hot mask that specifies the MCS index of each UE for each batch
        sample. Only required for training (if self._training==True).

    mcs_arr_eval : list with int elements
        Selects the elements (indices) of the mcs_index array to process.

    mcs_ue_mask_eval: [batch_size, max_num_tx, len(mcs_index)], tf.int32, None
        Optional additional parameter to specify an mcs_ue_mask for evaluation
        (self._training=False).
        Defaults to None, which internally assumes that all UEs are scheduled
        with mcs_arr_eval[0]

    Output
    ------
    Depending on the value of `training`:

    If `training` is set to `False`: (llr, h_hat_refined)

        llr : [batch_size, num_tx, num_data_symbols*num_bits_per_symbol],
              tf.float
            LLRs for every bit of every stream for specified MCS.

        h_hat_refined: [batch_size, num_tx, num_effective_subcarriers,
                    num_ofdm_symbols, 2*num_rx_ant]
            Refined channel estimate from the NRX.

    If `training` is set to `True`: (loss_data, loss_chest)

        loss_data: tf.float
            Binary cross-entropy loss on LLRs. Computed from active UEs and
            their selected MCSs.

        loss_chest: tf.float
            Mean-squared-error (MSE) loss between channel estimates and ground
            truth channel CFRs. Only relevant if double-readout is used.

    Note
    ----
    Receiver only supports single stream per user.

    """

    def __init__(self,
                 sys_parameters,
                 max_num_tx,
                 training,
                 num_it=5,
                 d_s=32,
                 num_units_init=[64],
                 num_units_agg=[[64]],
                 num_units_state=[[64]],
                 num_units_readout=[64],
                 layer_demappers=None,
                 layer_type_dense="dense",
                 layer_type_conv="sepconv",
                 layer_type_readout="dense",
                 nrx_dtype=tf.float32,
                 **kwargs):
        super().__init__(**kwargs)

        self._training = training
        self._max_num_tx = max_num_tx
        self._layer_demappers = layer_demappers
        self._sys_parameters = sys_parameters
        self._nrx_dtype = nrx_dtype

        self._num_mcss_supported = len(sys_parameters.mcs_index)

        self._rg = sys_parameters.transmitters[0]._resource_grid

        if self._sys_parameters.mask_pilots:
            print("Masking pilots for pilotless communications.")

        self._mcs_var_mcs_masking = False
        if hasattr(self._sys_parameters, 'mcs_var_mcs_masking'):
            self._mcs_var_mcs_masking = self._sys_parameters.mcs_var_mcs_masking
            print("Var-MCS NRX with masking.")
        elif len(sys_parameters.mcs_index) > 1:
            print("Var-MCS NRX with MCS-specific IO layers.")
        else:
            # Single-MCS NRX.
            pass

        # all UEs in the same pusch config must use the same MCS
        num_bits_per_symbol = []
        for mcs_list_idx in range(self._num_mcss_supported):
             num_bits_per_symbol.append(
                        sys_parameters.pusch_configs[mcs_list_idx][0].tb.num_bits_per_symbol)

        # Number of receive antennas
        num_rx_ant = sys_parameters.num_rx_antennas

        ####################################################
        # Core neural receiver
        ####################################################
        self._cgnn = CGNN(num_bits_per_symbol,  # is a list
                          num_rx_ant,
                          num_it,
                          d_s,
                          num_units_init,
                          num_units_agg,
                          num_units_state,
                          num_units_readout,
                          training=training,
                          layer_type_dense=layer_type_dense,
                          layer_type_conv=layer_type_conv,
                          layer_type_readout=layer_type_readout,
                          var_mcs_masking=self._mcs_var_mcs_masking,
                          dtype=nrx_dtype)

        ###################################################
        # Resource grid demapper to extract the
        # data-carrying resource elements from the
        # resource grid
        ###################################################
        self._rg_demapper = ResourceGridDemapper(self._rg,
                                                 sys_parameters.sm)

        #################################################
        # Instantiate the loss function if training
        #################################################
        if training:
            # Loss function
            self._bce = tf.keras.losses.BinaryCrossentropy(
                    from_logits=True,
                    reduction=tf.keras.losses.Reduction.NONE)
            # Loss function
            self._mse = tf.keras.losses.MeanSquaredError(
                reduction=tf.keras.losses.Reduction.NONE)

        ###############################################
        # Pre-compute positional encoding.
        # Positional encoding consists in the distance
        # to the nearest pilot in time and frequency.
        # It is therefore a 2D positional encoding.
        ##############################################

        # Indices of the pilot-carrying resource elements and pilot symbols
        rg_type = self._rg.build_type_grid()[:,0] # One stream only
        pilot_ind = tf.where(rg_type==1)
        pilots = flatten_last_dims(self._rg.pilot_pattern.pilots, 3)
        # Resource grid carrying only the pilots
        # [max_num_tx, num_effective_subcarriers, num_ofdm_symbols]
        pilots_only = tf.scatter_nd(pilot_ind, pilots,
                                    rg_type.shape)
        # Indices of pilots carrying RE (transmitter, freq, time)
        pilot_ind = tf.where(tf.abs(pilots_only) > 1e-3)
        pilot_ind = np.array(pilot_ind)

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
                                        self._rg.num_ofdm_symbols,
                                        self._rg.fft_size,
                                        pilot_ind_sorted.shape[1]])
        # Distance to the nearest pilot in frequency
        # Initialized with zeros and then filled
        pilots_dist_freq = np.zeros([   max_num_tx,
                                        self._rg.num_ofdm_symbols,
                                        self._rg.fft_size,
                                        pilot_ind_sorted.shape[1]])

        t_ind = np.arange(self._rg.num_ofdm_symbols)
        f_ind = np.arange(self._rg.fft_size)

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

    @property
    def num_it(self):
        """Number of receiver iterations. No weight sharing is used."""
        return self._cgnn.num_it

    @num_it.setter
    def num_it(self, val):
        self._cgnn.num_it = val

    def call(self, inputs, mcs_arr_eval, mcs_ue_mask_eval=None):

        # training requires to feed the inputs
        # mcs_ue_mask: [batch_size, num_tx, num_mcss], tf.float
        if self._training:
            y, h_hat_init, active_tx, bits, h, mcs_ue_mask = inputs
        else:
            y, h_hat_init, active_tx = inputs
            if mcs_ue_mask_eval is None:
                mcs_ue_mask = tf.one_hot(mcs_arr_eval[0],
                                         depth=self._num_mcss_supported)
            else:
                mcs_ue_mask = mcs_ue_mask_eval
            mcs_ue_mask = expand_to_rank(mcs_ue_mask, 3, axis=0)

        # total number of possible streams; not all of them might be active.
        num_tx = tf.shape(active_tx)[1]

        # mask pilots for pilotless communications
        if self._sys_parameters.mask_pilots:
            rg_type = self._rg.build_type_grid()
            # add batch dim
            rg_type = tf.expand_dims(rg_type, axis=0)
            rg_type = tf.broadcast_to(rg_type, tf.shape(y))
            y = tf.where(rg_type==1, tf.constant(0., y.dtype), y)

        ##############################################
        # Core Neural Receiver
        ##############################################

        # Reshaping to the expected shape
        # [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        y = y[:,0]
        y = tf.transpose(y, [0, 3, 2, 1])
        y = tf.concat([tf.math.real(y), tf.math.imag(y)], axis=-1)
        # Map showing the position of the nearest pilot for every user in time
        # and frequency.
        # This can be seen as a form of positional encoding
        # [num_tx, num_subcarriers, num_ofdm_symbols, 2]
        pe = self._nearest_pilot_dist[:num_tx]

        # Calling the detector to compute LLRs.
        # List of size num_it of tensors of LLRs with shape:
        # llrs : [batch_size, num_tx, num_effective_subcarriers,
        #       num_ofdm_symbols, num_bits_per_symbol]
        # h_hats : [batch_size, num_tx, num_effective_subcarriers,
        #       num_ofdm_symbols, 2*num_rx_ant]

        # cast to desired dtypes
        y = tf.cast(y, self._nrx_dtype)
        pe = tf.cast(pe, self._nrx_dtype)

        if h_hat_init is not None:
            h_hat_init = tf.cast(h_hat_init, self._nrx_dtype)
        active_tx = tf.cast(active_tx, self._nrx_dtype)

        # and run the neural receiver
        llrs_, h_hats_ = self._cgnn([y, pe, h_hat_init, active_tx, mcs_ue_mask])

        indices = mcs_arr_eval

        # list of lists; outer list separates iterations, inner list the MCSs
        llrs = []

        h_hats = []
        # process each list entry (=iteration) individually
        for llrs_, h_hat_ in zip(llrs_, h_hats_):

            h_hat_ = tf.cast(h_hat_, tf.float32)

            # local llrs list to only process LLRs of MCS indices specified in
            # mcs_arr_eval
            _llrs_ = []
            # loop over all evaluated mcs indices
            for idx in indices:

                # cast back to tf.float32 (if NRX uses quantization)
                llrs_[idx] = tf.cast(llrs_[idx], tf.float32)

                # llr : [batch_size, num_tx, num_effective_subcarriers,
                #       num_ofdm_symbols, num_bits_per_symbol]

                # Extract data-carrying REs
                # Reshape to
                # [batch_size, 1, num_tx, num_ofdm_symbols,
                #   fft_size, num_bits_per_symbol]
                llrs_[idx] = tf.transpose(llrs_[idx], [0, 1, 3, 2, 4])
                llrs_[idx] = tf.expand_dims(llrs_[idx], axis=1)
                # Need to pad LLRs as the RG demapper expects ``max_num_tx``
                # users
                # [batch_size, 1, max_num_tx, num_ofdm_symbols,
                #   fft_size, num_bits_per_symbol]
                #llr = tf.pad(llr, [ [0, 0], [0, 0],
                #                    [0, self._max_num_tx-num_tx],
                #                    [0,0], [0,0], [0, 0]])
                # [batch_size, num_tx, 1, num_data_symbols, num_bit_per_symbols]
                llrs_[idx] = self._rg_demapper(llrs_[idx])
                llrs_[idx] = llrs_[idx][:,:num_tx]

                # Keeping only the relevant users and the unique stream per user
                # [batch_size, num_tx, 1, num_data_symbols*num_bit_per_symbols]
                llrs_[idx] = flatten_last_dims(llrs_[idx], 2)

                # Remove stream dimension, NOTE: does not support
                # multiple-streams
                # per user; conceptually the neural receiver does, but would
                # require modified reshapes
                if self._layer_demappers is None:
                    llrs_[idx] = tf.squeeze(llrs_[idx], axis=-2)
                else:
                    llrs_[idx] = self._layer_demappers[idx](llrs_[idx])
                _llrs_.append(llrs_[idx])

            # llr is of shape
            # [batch_size, num_tx, num_data_symbols*num_bit_per_symbols]

            # h_hat is of shape
            # [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols,
            #   2*num_rx_ant]
            llrs.append(_llrs_)
            h_hats.append(h_hat_)

        if self._training:

            # Loss on data
            loss_data = tf.constant(0.0, dtype=tf.float32)
            for llrs_ in llrs:
                for idx in range(len(indices)):
                    loss_data_ = self._bce(bits[idx], llrs_[idx])

                    mcs_ue_mask_ = expand_to_rank(
                        tf.gather(mcs_ue_mask, indices=indices[idx], axis=2),
                        tf.rank(loss_data_), axis=-1)

                    # select data loss only for associated MCSs
                    loss_data_ = tf.multiply(loss_data_, mcs_ue_mask_)


                    # only focus on active users
                    active_tx_data = expand_to_rank(active_tx,
                                                    tf.rank(loss_data_),
                                                    axis=-1)
                    loss_data_ = tf.multiply(loss_data_, active_tx_data)
                    # Average over batch, transmitters, and resource grid
                    loss_data += tf.reduce_mean(loss_data_)

            # Loss on channel estimation
            loss_chest = tf.constant(0.0, dtype=tf.float32)
            if h_hats is not None: # h_hat might not be available
                for h_hat_ in h_hats:
                    if h is not None:
                        loss_chest += self._mse(h, h_hat_)

            # only focus on active users
            active_tx_chest = expand_to_rank(active_tx,
                                             tf.rank(loss_chest), axis=-1)
            loss_chest = tf.multiply(loss_chest, active_tx_chest)
            # Average over batch, transmitters, and resource grid
            loss_chest = tf.reduce_mean(loss_chest)
            return loss_data, loss_chest
        else:
            # Only return the last iteration during inference
            return llrs[-1][0], h_hats[-1]

class NeuralPUSCHReceiver(Layer):
    # pylint: disable=line-too-long
    r"""
    Neural PUSCH Receiver extending the CGNNOFDM Layer with 5G NR capabilities.

    This layer wraps the CGNNOFDM Layer such that it is 5G NR compatible.
    It includes all required steps for Transportblock (TB)/FEC decoding
    including scrambling and interleaving.

    Remark: for training, the labels are re-encoded with the TB-Encoder and
    thus the payload (transport block) information bits must be provided.
    In most practical use-cases this simplifies the data acquisition.

    Parameters
    ----------
    sys_parameters : Parameters
        The system parameters.

    training : boolean
        Set to `True` if instantiated for training. Set to `False` otherwise.

    Input
    ------
    (y, active_tx, [bits, h, mcs_ue_mask]) :
        Tuple: last two inputs are only for training mode

        y : [batch_size, num_rx, num_rx_ant, num_ofdm_symbols, fft_size], tf.complex
            The received OFDM resource grid after cyclic prefix removal and FFT.

        active_tx: [batch_size, num_tx], tf.float
            Active user mask where each `0` indicates non-active users and `1`
            indicates an active user.

        bits : list of [[batch_size, num_tx, num_data_symbols*num_bits_per_symbol],
                        tf.int]
            Transmitted information (uncoded) bits for each evaluated MCS.
            Only required for training to compute the loss function.

        h : [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            num_ofdm_symbols, fft_size], tf.complex
            Ground-truth channel impulse response.
            Only required for training to compute the loss function.

        mcs_ue_mask: [batch_size, max_num_tx, len(mcs_index)], tf.int32
            One-hot mask that specifies the MCS index of each UE for each batch
            sample. Only required for training to enable UE-specific MCS
            association.

    mcs_arr_eval : list with int elements
        Selects the elements (indices) of the mcs_index array to process.
        Defaults to [0]

    mcs_ue_mask_eval : [batch_size, max_num_tx, len(mcs_index)], tf.int32, None
        Optional additional parameter to specify an mcs_ue_mask for evaluation
        (self._training=False).
        Defaults to None, which internally assumes that all UEs are scheduled
        with mcs_arr_eval[0]

    Output
    ------
    Depending on the value of `training`:

    If Training set to `False`
        Inference only implemented for one MCS (first element in mcs_arr_eval)

    (b_hat, h_hat_refined, h_hat, tb_crc_status) : tuple

        b_hat : [batch_size, num_tx, tb_size], tf.float
            Reconstructed transport block bits after decoding.

        h_hat_refined, [batch_size, num_tx, num_effective_subcarriers,
                    num_ofdm_symbols, 2*num_rx_ant]
            Refined channel estimate from the NRX.

        h_hat, [batch_size, num_tx, num_effective_subcarriers,
                   num_ofdm_symbols, 2*num_rx_ant]
            Initial channel estimate used for the NRX.

        tb_crc_status: [batch_size, num_tx]
            Status of the TB CRC for each decoded TB.

    If Training set to `True`

    (loss_data, loss_chest) : tuple

        loss_data: tf.float
            Binary cross-entropy loss on LLRs. Computed from active UEs and
            their selected MCSs.

        loss_chest: tf.float
            Mean-squared-error (MSE) loss between channel estimates and ground
            truth channel CFRs. Only relevant if double-readout is used.
    """

    def __init__(self,
                sys_parameters,
                training=False,
                **kwargs):


        super().__init__(**kwargs)

        self._sys_parameters = sys_parameters

        self._training = training

        # init transport block enc/decoder
        self._tb_encoders = []   # @TODO encoderS and decoderS
        self._tb_decoders= []

        self._num_mcss_supported = len(sys_parameters.mcs_index)
        for mcs_list_idx in range(self._num_mcss_supported):
                self._tb_encoders.append(
                    self._sys_parameters.transmitters[mcs_list_idx]._tb_encoder)

                self._tb_decoders.append(
                    TBDecoder(self._tb_encoders[mcs_list_idx],
                              num_bp_iter=sys_parameters.num_bp_iter,
                              cn_type=sys_parameters.cn_type))

        # Precoding matrix to post-process the ground-truth channel when
        # training
        #  [num_tx, num_tx_ant, num_layers = 1]
        if hasattr(sys_parameters.transmitters[0], "_precoder"):
            self._precoding_mat = sys_parameters.transmitters[0]._precoder._w
        else:
            self._precoding_mat = tf.ones([sys_parameters.max_num_tx,
                                           sys_parameters.num_antenna_ports, 1], tf.complex64)

        # LS channel estimator
        # rg independent of MCS index
        rg = sys_parameters.transmitters[0]._resource_grid
        # get pc from first MCS and first Tx
        pc =  sys_parameters.pusch_configs[0][0]
        self._ls_est = PUSCHLSChannelEstimator(
                resource_grid=rg,
                dmrs_length=pc.dmrs.length,
                dmrs_additional_position=pc.dmrs.additional_position,
                num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
                interpolation_type="nn")

        rg_type = rg.build_type_grid()[:,0] # One stream only
        pilot_ind = tf.where(rg_type==1)
        self._pilot_ind = np.array(pilot_ind)

        # required to remove layers
        self._layer_demappers = []
        for mcs_list_idx in range(self._num_mcss_supported):
                self._layer_demappers.append(
                    LayerDemapper(
                            self._sys_parameters.transmitters[mcs_list_idx]._layer_mapper,
                            sys_parameters.transmitters[mcs_list_idx]._num_bits_per_symbol))

        self._neural_rx = CGNNOFDM(
                    sys_parameters,
                    max_num_tx=sys_parameters.max_num_tx,
                    training=training,
                    num_it=sys_parameters.num_nrx_iter,
                    d_s=sys_parameters.d_s,
                    num_units_init=sys_parameters.num_units_init,
                    num_units_agg=sys_parameters.num_units_agg,
                    num_units_state=sys_parameters.num_units_state,
                    num_units_readout=sys_parameters.num_units_readout,
                    layer_demappers=self._layer_demappers,
                    layer_type_dense=sys_parameters.layer_type_dense,
                    layer_type_conv=sys_parameters.layer_type_conv,
                    layer_type_readout=sys_parameters.layer_type_readout,
                    dtype=sys_parameters.nrx_dtype)

    def estimate_channel(self, y, num_tx):

        # y has shape
        #[batch_size, num_rx, num_rx_ant, num_ofdm_symbols, num_subcarriers]

        if self._sys_parameters.initial_chest == 'ls':
            if self._sys_parameters.mask_pilots:
                raise ValueError("Cannot use initial channel estimator if " \
                                "pilots are masked.")
            # [batch_size, num_rx, num_rx_ant, num_tx, num_streams_per_tx,
            #    num_ofdm_symbols, num_effective_subcarriers]
            # Dummy value for N0 as it is not used anyway.
            h_hat, _ = self._ls_est([y, 1e-1])

            # Reshaping to the expected shape
            # [batch_size, num_tx, num_effective_subcarriers,
            #       num_ofdm_symbols, 2*num_rx_ant]
            h_hat = h_hat[:,0,:,:num_tx,0]
            h_hat = tf.transpose(h_hat, [0, 2, 4, 3, 1])
            h_hat = tf.concat([tf.math.real(h_hat), tf.math.imag(h_hat)],
                              axis=-1)

        elif self._sys_parameters.initial_chest == None:
            h_hat = None

        return h_hat

    def preprocess_channel_ground_truth(self, h):
        # h : [batch_size, num_rx, num_rx_ant, num_tx, num_tx_ant,
        #       num_ofdm_symbols, num_effective_subcarriers]

        # Assume only one rx
        # [batch_size, num_rx_ant, num_tx, num_tx_ant, num_ofdm_symbols,
        #   fft_size]
        h = tf.squeeze(h, axis=1)

        # Reshape h
        # [batch size, num_tx, num_ofdm_symbols, num_effective_subcarriers,
        #   num_rx_ant, num_tx_ant]
        h = tf.transpose(h, perm=[0,2,5,4,1,3])

        # Multiply by precoding matrices to compute effective channels
        # [1, num_tx, 1, 1, num_tx_ant, 1]
        w = insert_dims(tf.expand_dims(self._precoding_mat, axis=0), 2, 2)
        # [batch size, num_tx, num_ofdm_symbols, num_effective_subcarriers,
        #   num_rx_ant]
        h = tf.squeeze(tf.matmul(h, w), axis=-1)

        # Complex-to-real
        # [batch size, num_tx, num_ofdm_symbols, num_effective_subcarriers,
        #   2*num_rx_ant]
        h = tf.concat([tf.math.real(h), tf.math.imag(h)], axis=-1)

        return h

    def call(self, inputs, mcs_arr_eval=[0], mcs_ue_mask_eval=None):
        """
        Apply neural receiver.
        """

        # assume u is provided as input in training mode
        if self._training:
            y, active_tx, b, h, mcs_ue_mask  = inputs
            # re-encode bits in training mode to generate labels
            # avoids the need for post-FEC bits as labels
            if len(mcs_arr_eval)==1 and not isinstance(b, list):
                b = [b] # generate new list if b is not provided as list
            bits = []
            for idx in range(len(mcs_arr_eval)):
                bits.append(
                    self._sys_parameters.transmitters[mcs_arr_eval[idx]]._tb_encoder(b[idx]))

            # Initial channel estimation
            num_tx = tf.shape(active_tx)[1]
            h_hat = self.estimate_channel(y, num_tx)

            # Reshaping `h` to the expected shape and apply precoding matrices
            # [batch size, num_tx, num_ofdm_symbols, num_effective_subcarriers,
            #   2*num_rx_ant]
            if h is not None:
                h = self.preprocess_channel_ground_truth(h)

            # Apply neural receiver and return loss
            losses = self._neural_rx((y, h_hat, active_tx,
                                      bits, h, mcs_ue_mask),
                                      mcs_arr_eval)
            return losses

        else:
            y, active_tx = inputs

            # Initial channel estimation
            num_tx = tf.shape(active_tx)[1]
            h_hat = self.estimate_channel(y, num_tx)

            llr, h_hat_refined = self._neural_rx(
                                            (y, h_hat, active_tx),
                                            [mcs_arr_eval[0]],
                                            mcs_ue_mask_eval=mcs_ue_mask_eval)

            # apply TBDecoding
            b_hat, tb_crc_status = self._tb_decoders[mcs_arr_eval[0]](llr)

            return b_hat, h_hat_refined, h_hat, tb_crc_status


################################
## ONNX Layers / Wrapper
################################
# The following layers provide an adapter to the Aerial PUSCH pipeline
# the code is only relevant for for ONNX/TensorRT exports but can be ignored
# for Sionna-based simulations.

class NRPreprocessing(Layer):
    # pylint: disable=line-too-long
    r"""
    Pre-preprocessing layer for the neural receiver applying initial channel
    estimation. In particular, the layer takes the channel estimates at pilot
    positions and performs nearest neighbor interpolation on a "per PRB" basis.
    As such it scales to arbitrary resource grid sizes.

    The input/output shapes are Aerial compatible and not directly compatible
    with Sionna. The returned "resourcegrid of LLRs" can be further processed
    using PyAerial.

    Note that all operations are real-valued.

    Parameters
    -----------
    num_tx : int
        Number of transmitters (i.e., independent streams)

    Input
    ------
    (y, h_hat, dmrs_ofdm_pos, dmrs_subcarrier_pos)
    Tuple:

    y : [batch_size, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant], tf.float32
        The received OFDM resource grid after cyclic prefix removal and FFT.
        Real and imaginary part are stacked in the rx_antenna direction.

    h_hat : [bs, num_pilots, num_streams, 2*num_rx_ant], tf.float32
        Channel estimates at pilot positions. Real and imaginary part are stacked in the rx_antenna direction.

    dmrs_ofdm_pos: [num_tx, num_dmrs_symbols], tf.int
        DMRS symbol positions within slot.

    dmrs_subcarrier_pos: [num_tx, num_pilots_per_PRB], tf.int
        Pilot position per PRB.

    Output
    -------
    [h_hat, pe]: Tuple

    h_hat : [batch_size, num_tx, num_subcarriers, num_ofdm_symbols,
             2*num_rx_ant], tf.float
        Channel estimate after nearest neighbor interpolation.

    pe : [num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
        Map showing the position of the nearest pilot for every user in time
        and frequency. This can be seen as a form of positional encoding.
    """

    def __init__(self,
                 num_tx,
                 **kwargs):

        super().__init__(**kwargs)

        self._num_tx = num_tx
        self._num_res_per_prb = 12 # fixed in 5G

    def _focc_removal(self, h_hat):
        """
        Apply FOCC removal to h_hat.

        Parameters
        ----------
        h_hat: [bs, 2*num_rx_ants, num_layers, num_pilots], tf.complex64
            Channel estimates at pilot positions.

        Outputs
        -------
        h_hat: [bs, 2*num_rx_ants, num_layers, num_pilots], tf.complex64
            Channel estimates at pilot positions after FOCC removal.
        """

        shape = [-1, 2]
        s = tf.shape(h_hat)
        new_shape = tf.concat([s[:3], shape], 0)
        # [bs, num_rx_ants, num_tx, num_pilots//2, 2]
        h_hat = tf.reshape(h_hat, new_shape)

        # [bs, num_rx_ants, num_tx, num_pilots//2, 1]
        h_hat = tf.reduce_sum(h_hat, axis=-1, keepdims=True) \
                                    / tf.cast(2., dtype=h_hat.dtype)
        # [bs, num_rx_ants, num_tx, num_pilots//2, 2]

        # we split into 2 as 4 was only required if we feed the zeroed/masked
        # pilots as well. This is not the case in the Aerial format
        # (only non-zero pilots are returned)
        h_hat = tf.repeat(h_hat, 2, axis=-1)

        # [bs, num_rx_ants, num_tx, num_pilots]
        shape = [-1]
        s = tf.shape(h_hat)
        new_shape = tf.concat([s[:3], shape], 0)
        h_ls = tf.reshape(h_hat, new_shape)

        return h_ls

    def _calculate_nn_indices(self, dmrs_ofdm_pos, dmrs_subcarrier_pos,
                              num_ofdm_symbols, num_prbs):
        """
        Calculates nearest neighbor interpolation indices for a single PRB.

        Parameters
        ----------
        dmrs_ofdm_pos: [num_tx, num_dmrs_symbols], tf.int
            DMRS symbol position within slot.

        dmrs_subcarrier_pos: [num_tx, num_pilots_per_PRB], tf.int
            Pilot position per PRB.

        num_ofdm_symbol: tf.int
            Number of OFDM symbols per slot.

        num_prbs: tf.int
            Number of allocated PRBs.

        Output
        ------
        nn_idx : [num_tx, num_streams_per_tx, num_ofdm_symbols,
                  num_effective_subcarriers=12], tf.int32
            Indices of nearest pilot RG grid. Can be used for nearest neighbor
            interpolation.

        pe : [num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
            Map showing the position of the nearest pilot for every user in time
            and frequency. This can be seen as a form of positional encoding.
        """

        re_pos = tf.meshgrid(tf.range(self._num_res_per_prb),
                             tf.range(num_ofdm_symbols))
        re_pos = tf.stack(re_pos, axis=-1)
        # enable broadcasting for distance calculation
        re_pos = tf.reshape(re_pos, (-1,1,2))

        pes = []
        nn_idxs = []
        # calculate distance per tx
        for tx_idx in range(self._num_tx):
            # Combining the coordinates into a single matrix
            p_idx= tf.meshgrid(dmrs_subcarrier_pos[tx_idx],
                               dmrs_ofdm_pos[tx_idx])
            pilot_pos = tf.stack(p_idx, axis=-1)
            # Reshaping to get a list of coordinate pairs
            pilot_pos = tf.reshape(pilot_pos, (-1, 2))

            pilot_pos = tf.reshape(pilot_pos, (1,-1,2))
            diff = tf.abs(re_pos - pilot_pos)
            # Manhattan distance
            dist = tf.reduce_sum(diff, axis=-1)

            # find indices of closes pilot
            nn_idx = tf.argmin(dist, axis=1)

            # and bring into shape [num_tx, num_streams_per_tx,...
            # ... num_ofdm_symbols,num_effective_subcarriers]
            nn_idx = tf.reshape(nn_idx,
                                (1, 1, num_ofdm_symbols, self._num_res_per_prb))

            # [num_tx, num_subcarriers, num_ofdm_symbols, 2],
            pe = tf.reduce_min(diff, axis=1)
            pe = tf.reshape(pe,
                            (1, num_ofdm_symbols, self._num_res_per_prb, 2))
            pe = tf.transpose(pe, (0,2,1,3))

            # normalize per axis(t and f)
            # remark leads to effect that in f dim, we have +/-1
            p = []
            pe = tf.cast(pe, tf.float32)

            # ofdm symbol axis
            pe_ = pe[...,1:2]
            pe_ -= tf.reduce_mean(pe_)#,axis=2, keepdims=True)
            std_ = tf.math.reduce_std(pe_)#,axis=2, keepdims=True)
            pe_ = tf.where(std_>0., pe_/std_, pe_)
            p.append(pe_)

            # subcarrier axis
            pe_ = pe[...,0:1]
            pe_ -= tf.reduce_mean(pe_)#,axis=1, keepdims=True)
            std_ = tf.math.reduce_std(pe_)#,axis=1, keepdims=True)
            pe_ = tf.where(std_>0., pe_/std_, pe_)
            p.append(pe_)

            pe = tf.concat(p ,axis=-1)

            pes.append(pe)
            nn_idxs.append(nn_idx)

        pe = tf.concat(pes, axis=0)
        pe = tf.tile(pe, (1, num_prbs, 1, 1)) # broadcasting over all PRBs
        nn_idx = tf.concat(nn_idxs, axis=0)
        nn_idx = tf.concat(nn_idxs, axis=0)
        return nn_idx, pe

    def _nn_interpolation(self, h_hat, num_ofdm_symbols,dmrs_ofdm_pos,
                          dmrs_subcarrier_pos):
        """
        Applies nearest neighbor interpolation of pilots to all data
        symbols in the resource grid.

        Remark: NN interpolation is done per PRB.

        Parameters
        ----------
        h_hat: [bs, 2*num_rx_ants, num_layers, num_pilots], tf.complex
            Channel estimates at pilot locations

        num_ofdm_symbols: tf.int
            Total number of OFDM symbols per slot.

        dmrs_ofdm_pos: [num_tx, num_dmrs_symbols], tf.int
            DMRS symbol position within slot.

        dmrs_subcarrier_pos: [num_tx, num_pilots_per_PRB], tf.int
            Pilot position per PRB.

        Output
        ------
        h_hat: [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols,
                num_effective_subcarriers], tf.complex
            Interpolated channel estimates of the entire resource grid.

        pe : [num_tx, num_subcarriers, num_ofdm_symbols, 2], tf.float
            Map showing the position of the nearest pilot for every user in time
            and frequency. This can be seen as a form of positional encoding.
        """
        # derive shapes from h_hat (and not y) as TRT has issues otherwise with
        # dynamic input shapes
        num_pilots_per_dmrs = tf.shape(dmrs_subcarrier_pos)[1] # pilot symbols
        num_prbs = tf.cast(tf.shape(h_hat)[-1]
                        / (num_pilots_per_dmrs * tf.shape(dmrs_ofdm_pos)[-1]),
                           tf.int32)

        # permute input estimates such that the first half of pilots of the
        # first PRB follows its second half (or third if multiple dmrs symbols)
        s = tf.shape(h_hat)
        h_hat = split_dim(h_hat, shape=(-1, num_pilots_per_dmrs), axis=3)
        h_hat = split_dim(h_hat, shape=(-1, num_prbs), axis=3)
        h_hat = tf.transpose(h_hat, (0,1,2,4,3,5))
        # flatten dims does not work with ONNX (partially unknown shapes)
        # h_hat = flatten_dims(h_hat, 3, axis=3)
        h_hat = tf.reshape(h_hat, s)

        # bring into Sionna compatible shape
        h_hat = tf.expand_dims(h_hat, axis=1) # num_rx
        h_hat = tf.expand_dims(h_hat, axis=4) # num_streams_per_tx
        perm = tf.roll(tf.range(tf.rank(h_hat)), -3, 0)
        h_hat = tf.transpose(h_hat, perm)

        # compute indices on the fly
        ls_nn_ind, pe = self._calculate_nn_indices(dmrs_ofdm_pos,
                                                   dmrs_subcarrier_pos,
                                                   num_ofdm_symbols,
                                                   num_prbs)

        # reshape such that interpolation is applied to each PRB individually
        s = tf.shape(h_hat)
        h_hat_prb = split_dim(h_hat, shape=(num_prbs, -1), axis=2)
        h_hat_prb = tf.transpose(h_hat_prb, (0,1,3,2,4,5,6))
        # apply nn interpolation
        outputs = tf.gather(h_hat_prb, ls_nn_ind, 2, batch_dims=2)
        outputs = tf.transpose(outputs, (0,1,2,4,3,5,6,7))

        # and remove artificial "per-prb" dimension
        s = tf.shape(outputs)
        s = tf.concat((tf.constant((-1,), tf.int32), s[1:3],
                       tf.expand_dims(num_prbs*self._num_res_per_prb, axis=0),
                       s[5:]), axis=0)
        # and combine all PRBs
        outputs = tf.reshape(outputs, s)

        # Transpose outputs to bring batch_dims first again. New shape:
        # [k, l, m, num_tx, num_streams_per_tx, num_ofdm_symbols,
        #  num_effective_subcarriers]
        perm = tf.roll(tf.range(tf.rank(outputs)), 3, 0)
        h_hat = tf.transpose(outputs, perm)
        return h_hat, pe

    def call(self, inputs):

        y, h_hat_ls, dmrs_ofdm_pos, dmrs_subcarrier_pos = inputs

        num_ofdm_symbols = tf.shape(y)[2]

        # shape y [bs, num_subcarriers, num_ofdm_symbols, 2*num_rx_ant]
        # shape h_hat_ls [bs, num_pilots, num_layers, 2*num_rx_ants]
        # desired shape of h_hat_ls [bs, 2*num_rx_ants, num_layers, num_pilots]
        h_hat_ls = tf.transpose(h_hat_ls, (0,3,2,1))

        ######################
        ### FOCC removal
        ######################

        h_hat_ls = self._focc_removal(h_hat_ls)

        ######################
        ### NN interpolator
        ######################
        # Interpolate channel estimates over the RG
        h_hat, pe = self._nn_interpolation(h_hat_ls,
                                           num_ofdm_symbols,
                                           dmrs_ofdm_pos,
                                           dmrs_subcarrier_pos)

        # Reshaping to the expected shape
        # [batch_size, num_tx, num_effective_subcarriers,
        #       num_ofdm_symbols, 2*num_rx_ant]
        h_hat = h_hat[:,0,:,:self._num_tx,0]
        h_hat = tf.transpose(h_hat, [0, 2, 4, 3, 1])
        return [h_hat, pe]

class NeuralReceiverONNX(Model):
    # pylint: disable=line-too-long
    r"""
    Wraps the 5G NR neural receiver in an ONNX compatible format.

    Note that the shapes are optimized for Aerial and not directly compatible
    with Sionna.

    Parameters
    -----------
    num_it : int
        Number of iterations.

    d_s : int
        Size of the state vector.

    num_units_init : list of int
        Number of hidden units for the init network.

    num_units_agg : list of int
        Number of kernel for the hidden dense layers of the aggregation network

    num_units_state : list of int
        Number of hidden units for the state-update network.

    num_units_readout : list of int
        Number of hidden units for the read-out network.

    num_bits_per_symbol: int
        Number of bits per symbol.

    num_tx: int
        Max. number of layers/DMRS ports. Note that DMRS ports can be
        dynamically deactivated via the dmrs_port_mask.

    Input
    ------
    (rx_slot, h_hat, dmrs_port_mask)
    Tuple:

    rx_slot_real : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant],
                    tf.float32
        Real part of the received OFDM resource grid after cyclic prefix
        removal and FFT.

    rx_slot_imag : [batch_size, num_subcarriers, num_ofdm_symbols, num_rx_ant],
                    tf.float32
        Imaginary part of the received OFDM resource grid after cyclic prefix
        removal and FFT.

    h_hat_real : [batch_size, num_pilots, num_streams, num_rx_ant], tf.float32
        Real part of the LS channel estimates at pilot positions.

    h_hat_imag : [batch_size, num_pilots, num_streams, num_rx_ant], tf.float32
        Imaginary part of the LS channel estimates at pilot positions.

    dmrs_port_mask: [bs, num_tx], tf.float32
        Mask of 0s and 1s to indicate that DMRS ports are active or not.

    dmrs_ofdm_pos: [num_tx, num_dmrs_symbols], tf.int
        DMRS symbol positions within slot.

    dmrs_subcarrier_pos: [num_tx, num_pilots_per_prb], tf.int
        Pilot positions per PRB.

    Output
    -------
    (llr, h_hat)
    Tuple:

    llr : [batch_size, num_bits_per_symbol, num_tx, num_effective_subcarriers,
          num_ofdm_symbols], tf.float
        LLRs on bits.

    h_hat : [batch_size, num_tx, num_effective_subcarriers, num_ofdm_symbols,
             2*num_rx_ant], tf.float
        Refined channel estimates.
    """

    def __init__(self,
                 num_it,
                 d_s,
                 num_units_init,
                 num_units_agg,
                 num_units_state ,
                 num_units_readout,
                 num_bits_per_symbol,
                 layer_type_dense,
                 layer_type_conv,
                 layer_type_readout,
                 nrx_dtype,
                 num_tx,
                 num_rx_ant,
                 **kwargs):

        super().__init__(**kwargs)
        assert len(num_units_agg) == num_it and len(num_units_state) == num_it

        # hard-coded for simplicity
        self._num_tx = num_tx # we assume 1 stream per user

        ####################################################
        # Detector
        ####################################################
        self._cgnn = CGNN([num_bits_per_symbol], # no support for mixed MCS
                          num_rx_ant,
                          num_it,
                          d_s,
                          num_units_init,
                          num_units_agg,
                          num_units_state,
                          num_units_readout,
                          layer_type_dense=layer_type_dense,
                          layer_type_conv=layer_type_conv,
                          layer_type_readout=layer_type_readout,
                          dtype=nrx_dtype)

        self._preprocessing = NRPreprocessing(self._num_tx)

    @property
    def num_it(self):
        return self._num_it

    @num_it.setter
    def num_it(self, val):
        assert (val >= 1) and (val <= len(self._iterations)),\
            "Invalid number of iterations"
        self._num_it = val

    def call(self, inputs):

        y_real, y_imag, h_hat_real, h_hat_imag, \
            dmrs_port_mask, dmrs_ofdm_pos, dmrs_subcarrier_pos = inputs

        y = tf.concat((y_real, y_imag), axis=-1)
        h_hat_p = tf.concat((h_hat_real, h_hat_imag), axis=-1)

        # nearest neighbor interpolation of channel estimates
        h_hat, pe = self._preprocessing((y,
                                         h_hat_p,
                                         dmrs_ofdm_pos,
                                         dmrs_subcarrier_pos))

        # dummy MCS mask (no support for mixed MCS)
        mcs_ue_mask = tf.ones((1,1,1), tf.float32)

        # and run NRX
        llr, h_hat = self._cgnn([y, pe, h_hat, dmrs_port_mask, mcs_ue_mask])

        # cgnn returns list of results for each iteration
        # (not needed for inference)
        llr = llr[-1][0] # take LLRs of first MCS (no support for mixed MCS)
        h_hat = h_hat[-1]

        # cast back to tf.float32 (if NRX uses quantization)
        llr = tf.cast(llr, tf.float32)
        h_hat = tf.cast(h_hat, tf.float32)

        # reshape llrs in Aerial format
        llr = tf.transpose(llr, (0,4,1,2,3))
        # Sionna defines LLRs with different sign
        llr = -1. * llr

        return llr, h_hat
