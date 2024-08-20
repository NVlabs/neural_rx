# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

##### Hardware impairment layers

from tensorflow.keras.layers import Layer
import tensorflow as tf
from sionna.ofdm import OFDMModulator, OFDMDemodulator
from sionna.constants import PI

class FrequencyOffset(Layer):
    """
    FrequencyOffset(max_rel_offset, input_domain, resource_grid=None, constant_offset=False, **kwargs)

    Applies a random frequency offset to the input signal.

    Parameters
    ----------
    max_rel_offset: float
        Maximum absolute frequency offset relative to the sampling rate.

    input_domain: ["time", "freq"], str
        Domain of the input signal. It can be either "time" or "freq".

    resource_grid: object, optional
        Instance of resource grid. Only used if `input_domain` is "freq".
        Defaults to `None`.

    constant_offset: bool, optional
        If True, the frequency offset is kept constant. Otherwise,
        random frequency offsets are uniformly sampled in the range of
        `[-max_rel_offset, max_rel_offset]`. Defaults to `False`.

    Input
    -----
    x: [batch_size, num_tx, num_tx_ant, num_time_samples] or [batch_size,
        num_tx, num_tx_ant, num_ofdm_symbols, fft_size], tf.complex
        The signal to which the frequency offset should be applied.

    Output
    ------
    y: [batch_size, num_tx, num_tx_ant, num_time_samples], tf.complex
        The signal with the frequency offset applied.
    """

    def __init__(self, max_rel_offset, input_domain, resource_grid=None,
                 constant_offset=False, **kwargs):
        super().__init__(**kwargs)
        self._max_rel_offset = tf.cast(max_rel_offset, tf.float32)

        # Determine the range for random uniform offsets
        if constant_offset:
            self._min_rel_offset = self._max_rel_offset
        else:
            self._min_rel_offset = -self._max_rel_offset

        self._input_domain = input_domain
        self._resource_grid = resource_grid

        # Initialize OFDM modulator and demodulator if the input domain is
        # "freq"
        if self._input_domain == "freq":
            assert self._resource_grid is not None, \
                "resource_grid must be provided when input_domain is 'freq'."
            self._modulator = OFDMModulator(resource_grid.cyclic_prefix_length)
            self._demodulator = OFDMDemodulator(
                                    resource_grid.fft_size, 0,
                                    resource_grid.cyclic_prefix_length)

    def call(self, inputs):
        # If input domain is "freq", convert inputs to time domain
        # and apply CFO in time domain and convert back afterwards
        if self._input_domain == "freq":
            inputs = self._modulator(inputs)

        # Determine the number of time samples
        num_time_samples = tf.shape(inputs)[-1]

        # Create shape tensor for frequency offsets
        s = tf.concat((tf.shape(inputs)[0:2], tf.ones((2,), tf.int32)), axis=0)

        # Sample random frequency offsets
        fo = tf.random.uniform(s, # Shape: [batch_size, num_tx, 1, 1]
                               minval=self._min_rel_offset,
                               maxval=self._max_rel_offset,
                               dtype=tf.float32)

        # Calculate phase increments and shifts
        phase_increment = fo * 2 * PI
        time_steps = tf.reshape(
                            tf.range(0, num_time_samples, dtype=tf.float32),
                            [1, 1, 1, -1])
        phase_shifts = time_steps * phase_increment

        # Apply frequency offset to the input signal
        exp = tf.cast(tf.exp(tf.complex(0., phase_shifts)), inputs.dtype)
        outputs = exp * inputs

        # If input domain was "freq", convert outputs back to frequency domain
        if self._input_domain == "freq":
            outputs = self._demodulator(outputs)

        return outputs
