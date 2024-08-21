# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# reads in the configuration file and initialized all relevant system components
# This allows to train and evaluate different system configurations on the same # server and simplifies logging of the training process.

import numpy as np
import configparser
import tensorflow as tf
from os.path import exists
from sionna.nr import PUSCHConfig, PUSCHDMRSConfig, TBConfig, CarrierConfig, PUSCHTransmitter, PUSCHPilotPattern
from sionna.channel.tr38901 import PanelArray, UMi, TDL, UMa
from sionna.mimo import StreamManagement
from sionna.channel import OFDMChannel, AWGN
from .channel_models import DoubleTDLChannel, DatasetChannel
from .impairments import FrequencyOffset

class Parameters:
    r"""
    Simulation parameters

    Parameters
    ----------
    config_name : str
        name of the config file.

    system : str
        Receiver algorithm to be used.Must be one of:
        * "nrx" : Neural receiver
        * "baseline_lmmse_kbest" : LMMSE estimation and K-Best detection
        * "baseline_perf_csi_kbest" : perfect CSI and K-Best detection
        * "baseline_lmmse_lmmse" : LMMSE estimation and LMMSE equalization
        * "baseline_lsnn_lmmse" : LS estimation/nn interpolation and LMMSE equalization
        * "dummy" : stops after parameter import. Can be used only to parse the
        config.

    training: bool, False,
        If True, training parameters are loaded. Otherwise, the evaluation
        parameters are used.

    verbose: bool, False
        If True, additional information is printed during init.

    compute_cov: bool, False
        If True, the UMi channel model is loaded automatically to avoid
        overfitting to TDL models.

    num_tx_eval: int or None
        If provided, the max number of users is limited to ``num_tx_eval``.
        For this, the first DMRS ports are selected.
    """
    def __init__(self, config_name, system="dummy", training=False, verbose=False, compute_cov=False, num_tx_eval=None):

        # check input for consistency
        assert isinstance(verbose, bool), "verbose must be bool."
        assert isinstance(training, bool), "training must be bool."
        assert isinstance(config_name, str), "config_name must be str."
        assert isinstance(system, str), "system must be str."
        assert isinstance(compute_cov, bool), "compute_cov must be bool."

        self.system = system

        ###################################
        ##### Load configuration file #####
        ###################################

        # create parser object and read config file
        fn = f'../config/{config_name}'
        if exists(fn):
            config = configparser.RawConfigParser()
            # automatically add fileformat if needed
            config_name.replace(".cfg","") + ".cfg"
            config.read(fn)
        else:
            raise FileNotFoundError("Unknown config file.")

        # and import all parameters as attributes
        self.config_str = ""
        for section in config.sections():
            s = f"\n---- {section} ----- "
            self.config_str += s + "<br />" # add linebreak for Tensorboard
            if verbose:
                print(s)
            for option in config.options(section):
                setattr(self, f"{option}", eval(config.get(section,option)))
                s = f"{option}: {eval(config.get(section,option))}"
                self.config_str += s + "<br />" # add linebreak for Tensorboard
                if verbose:
                    print(s)

        # Overwrite channel and PRBs in inference mode with "eval" parameters
        # This allows to configure different parameters during training and
        # evaluation.
        if not training:
            self.channel_type = self.channel_type_eval
            self.n_size_bwp = self.n_size_bwp_eval
            self.max_ut_velocity = self.max_ut_velocity_eval
            self.min_ut_velocity = self.min_ut_velocity_eval
            self.channel_norm = self.channel_norm_eval
            self.cfo_offset_ppm = self.cfo_offset_ppm_eval
            self.tfrecord_filename = self.tfrecord_filename_eval
            if self.channel_type == "Dataset":
                self.random_subsampling = self.random_subsampling_eval

        # load only config parameters and return without initializing the rest
        # of the system
        if self.system == "dummy":
            return

        #####################################
        ##### Init PUSCH configurations #####
        #####################################

        # init PUSCHConfig
        carrier_config = CarrierConfig(
                n_cell_id=self.n_cell_id,
                cyclic_prefix=self.cyclic_prefix,
                subcarrier_spacing=int(self.subcarrier_spacing/1e3), # in kHz
                n_size_grid=self.n_size_bwp,
                n_start_grid=self.n_start_grid,
                slot_number=self.slot_number,
                frame_number=self.frame_number)

        # init DMRSConfig
        pusch_dmrs_config=PUSCHDMRSConfig(
                config_type=self.dmrs_config_type,
                type_a_position=self.dmrs_type_a_position,
                additional_position=self.dmrs_additional_position,
                length=self.dmrs_length,
                dmrs_port_set=self.dmrs_port_sets[0], # first user
                n_scid=self.n_scid,
            num_cdm_groups_without_data=self.num_cdm_groups_without_data)

        mcs_list = self.mcs_index
        # generate pusch configs for all MCSs
        self.pusch_configs = []   # self.pusch_configs[MCS_CONFIG][N_UE]
        for mcs_list_idx in range(len(mcs_list)):
            self.pusch_configs.append([])
            mcs_index = mcs_list[mcs_list_idx]
            # init TBConfig
            tb_config = TBConfig(
                    mcs_index=mcs_index,
                    mcs_table=self.mcs_table,
                    channel_type="PUSCH")
                    #n_id=self.n_ids[0])

            # first user PUSCH config
            pc = PUSCHConfig(
                    carrier_config=carrier_config,
                    pusch_dmrs_config=pusch_dmrs_config,
                    tb_config=tb_config,
                    num_antenna_ports = self.num_antenna_ports,
                    precoding = self.precoding,
                    symbol_allocation = self.symbol_allocation,
                    tpmi = self.tpmi,
                    mapping_type=self.dmrs_mapping_type,)

            # clone new PUSCHConfig for each additional user
            for idx,_ in enumerate(self.dmrs_port_sets):
                p = pc.clone() # generate new PUSCHConfig
                # set user specific parts
                p.dmrs.dmrs_port_set = self.dmrs_port_sets[idx]
                # The following parameters are derived from default.
                # Comment lines if specific configuration is not required.
                p.n_id = self.n_ids[idx]
                p.dmrs.n_id = self.dmrs_nid[idx]
                p.n_rnti = self.n_rntis[idx]
                self.pusch_configs[mcs_list_idx].append(p)

        ##############################
        ##### Consistency checks #####
        ##############################

        # after training we can only reduce the number of iterations
        assert self.num_nrx_iter_eval<=self.num_nrx_iter, \
            "num_nrx_iter_eval must be smaller or equal num_nrx_iter."

        # for the evaluation, only activate num_tx_eval configs
        if not training:
                # overwrite num_tx_eval if explicitly provided:
            if num_tx_eval is not None:
                num_tx_eval = num_tx_eval
            else: # if not provided use all available port sets
                num_tx_eval = len(self.dmrs_port_sets)
            self.max_num_tx = num_tx_eval # non-varying users for evaluation
            self.min_num_tx = num_tx_eval # non-varying users for evaluation

        for mcs_list_idx in range(len(mcs_list)):
            self.pusch_configs[mcs_list_idx] = self.pusch_configs[mcs_list_idx][:self.max_num_tx]
        print(f"Evaluating the first {self.max_num_tx} port sets.")

        ##################################
        ##### Configure Transmitters #####
        ##################################

        # Generate and store DMRS for all slot numbers
        self.pilots = []
        for slot_num in range(carrier_config.num_slots_per_frame):
            for pcs in self.pusch_configs:
                for pc in pcs:
                    pc.carrier.slot_number = slot_num
            # only generate pilot pattern for first MCS's PUSCH config, as
            # pilots are independent from MCS index
            pilot_pattern = PUSCHPilotPattern(self.pusch_configs[0])
            self.pilots.append(pilot_pattern.pilots)
        self.pilots = tf.stack(self.pilots, axis=0)
        self.pilots = tf.constant(self.pilots)
        for pcs in self.pusch_configs:
            for pc in pcs:
                pc.carrier.slot_number = self.slot_number

        # transmitter is a list of PUSCHTransmitters, one for each MCS
        self.transmitters = []
        for mcs_list_idx in range(len(mcs_list)):
            # and init transmitter
            self.transmitters.append(
                PUSCHTransmitter(
                            self.pusch_configs[mcs_list_idx],
                            return_bits=False,
                            output_domain="freq",
                            verbose=self.verbose))

            # support end-to-end learning / custom constellations
            # see https://arxiv.org/pdf/2009.05261 for details
            if self.custom_constellation: # trainable constellations
                print("Activating trainable custom constellations.")
                self.transmitters[mcs_list_idx]._mapper.constellation.trainable = True
            # Center constellations. This could be also deactivated for more
            # degrees of freedom.
            self.transmitters[mcs_list_idx]._mapper.constellation.center = True

        # chest will fail if we use explicit masking of pilots.
        if self.mask_pilots and self.initial_chest in ("ls", "nn"):
            print("Warning: initial_chest will fail with masked pilots.")

        # StreamManagement required for KBestDetector
        self.sm = StreamManagement(np.ones([1, self.max_num_tx], int), 1)

        ##############################
        ##### Initialize Channel #####
        ##############################

        # always use UMi to calculate covariance matrix
        if compute_cov:
            if not self.channel_type in ("UMi", "UMa"): # use UMa if selected
                print("Setting channel type to UMi for covariance computation.")
                self.channel_type = "UMi"

        # Sanity check
        if self.channel_type in ("DoubleTDLlow","DoubleTDLmedium",
                                 "DoubleTDLhigh") and self.max_num_tx==1:
                print("Warning: SelectedDoubleTDL model only defined for 2 "\
                      "users. Selecting TDL-B100 instead.")
                self.channel_type = "TDL-B100"

        # Initialize channel
        # Remark: new channel models can be added here
        if self.channel_type in ("UMi", "UMa"):
            if self.num_rx_antennas==1: # ignore polarization for single antenna
                print("Using vertical polarization for single antenna setup.")
                num_cols_per_panel = 1
                num_rows_per_panel = 1
                polarization = "single"
                polarization_type = 'V'
            else:
                # we use a ULA array to be aligned with TDL models
                num_cols_per_panel = self.num_rx_antennas//2
                num_rows_per_panel = 1
                polarization = "dual"
                polarization_type = 'cross'

            bs_array = PanelArray(num_rows_per_panel = num_rows_per_panel,
                                  num_cols_per_panel = num_cols_per_panel,
                                  polarization = polarization,
                                  polarization_type  = polarization_type,
                                  antenna_pattern = '38.901',
                                  carrier_frequency = self.carrier_frequency)

            ut_array = PanelArray(num_rows_per_panel = 1,
                                  num_cols_per_panel = pc.num_antenna_ports,
                                  polarization = 'single',
                                  polarization_type = 'V',
                                  antenna_pattern = 'omni',
                                  carrier_frequency = self.carrier_frequency)

            if self.channel_type == "UMi":
                self.channel_model = UMi(
                                carrier_frequency=self.carrier_frequency,
                                o2i_model = 'low',
                                bs_array = bs_array,
                                ut_array = ut_array,
                                direction = 'uplink',
                                enable_pathloss = False,
                                enable_shadow_fading = False)
            else: # UMa
                self.channel_model = UMa(
                                carrier_frequency=self.carrier_frequency,
                                o2i_model = 'low',
                                bs_array = bs_array,
                                ut_array = ut_array,
                                direction = 'uplink',
                                enable_pathloss = False,
                                enable_shadow_fading = False)

            self.channel = OFDMChannel(
                    channel_model=self.channel_model,
                    resource_grid=self.transmitters[0]._resource_grid,          # resource grid is independent of MCS
                    add_awgn=True,
                    normalize_channel=self.channel_norm,
                    return_channel=True)

        elif self.channel_type == "TDL-B100":
            tdl = TDL(model="B100",
                      delay_spread=100e-9,
                      carrier_frequency=self.carrier_frequency,
                      min_speed=self.min_ut_velocity,
                      max_speed=self.max_ut_velocity,
                      num_tx_ant=pc.num_antenna_ports,
                      num_rx_ant=self.num_rx_antennas)
            self.channel = OFDMChannel(tdl,
                                       self.transmitters[0].resource_grid,      # resource grid is independent of MCS
                                       add_awgn=True,
                                       normalize_channel=self.channel_norm,
                                       return_channel=True)
        elif self.channel_type == "TDL-C300":
            tdl = TDL(model="C300",
                      delay_spread=300e-9,
                      carrier_frequency=self.carrier_frequency,
                      min_speed=self.min_ut_velocity,
                      max_speed=self.max_ut_velocity,
                      num_tx_ant=pc.num_antenna_ports,
                      num_rx_ant=self.num_rx_antennas)
            self.channel = OFDMChannel(tdl,
                                       self.transmitters[0].resource_grid,      # resource grid is independent of MCS
                                       add_awgn=True,
                                       normalize_channel=self.channel_norm,
                                       return_channel=True)
        # DoubleTDL for evaluation
        elif self.channel_type == "DoubleTDLlow":
            self.channel = DoubleTDLChannel(self.carrier_frequency,
                                    self.transmitters[0].resource_grid,         # resource grid is independent of MCS
                                    correlation="low",
                                    num_tx_ant=pc.num_antenna_ports,
                                    norm_channel=self.channel_norm)
        # DoubleTDL for evaluation
        elif self.channel_type == "DoubleTDLmedium":
            self.channel= DoubleTDLChannel(self.carrier_frequency,
                                    self.transmitters[0].resource_grid,         # resource grid is independent of MCS
                                    correlation="medium",
                                    num_tx_ant=pc.num_antenna_ports,
                                    norm_channel=self.channel_norm)
        # DoubleTDL for evaluation
        elif self.channel_type == "DoubleTDLhigh":
            self.channel = DoubleTDLChannel(self.carrier_frequency,
                                    self.transmitters[0].resource_grid,         # resource grid is independent of MCS
                                    correlation="high",
                                    num_tx_ant=pc.num_antenna_ports,
                                    norm_channel=self.channel_norm)

        elif self.channel_type == "AWGN":
            self.channel = AWGN()


        elif self.channel_type == "Dataset":
            channel_model = DatasetChannel("../data/" + self.tfrecord_filename,
                                    max_num_examples=-1, # loads entire dataset
                                    training=training,
                                    num_tx=self.max_num_tx,
                                    random_subsampling=self.random_subsampling,
                                    )
            self.channel = OFDMChannel(channel_model,
                                       self.transmitters[0].resource_grid,      # resource grid is independent of MCS
                                       add_awgn=True,
                                       normalize_channel=self.channel_norm,
                                       return_channel=True)

        else:
            raise ValueError("Unknown Channel type.")

        # Hardware impairments
        if self.cfo_offset_ppm>0:
            offset = self.carrier_frequency / 1e6 * self.cfo_offset_ppm
            max_rel_offset = offset/self.transmitters[0].resource_grid.bandwidth    # resource grid and bandwidth is independent of MCS
            self.frequency_offset = FrequencyOffset(
                                    max_rel_offset,
                                    "freq",
                                    self.transmitters[0].resource_grid,             # resource grid is independent of MCS
                                    constant_offset=(not training)) # fix offset for evaluation
        else:
            self.frequency_offset = None


        ################
        ##### MISC #####
        ################

        # Load covariance matrices
        if self.system in ("baseline_lmmse_kbest", "baseline_lmmse_lmmse"):

            # test if files exist
            fn = f'../weights/{self.label}_time_cov_mat.npy'
            if not exists(fn):
                raise FileNotFoundError("time_cov_mat.npy not found. " \
                    "Please run compute_cov_mat.py for given config first.")

            self.space_cov_mat = tf.cast(np.load(
                        f'../weights/{self.label}_space_cov_mat.npy'),
                                                tf.complex64)
            self.time_cov_mat = tf.cast(np.load(
                        f'../weights/{self.label}_time_cov_mat.npy'),
                                            tf.complex64)
            self.freq_cov_mat = tf.cast(np.load(
                        f'../weights/{self.label}_freq_cov_mat.npy'),
                                            tf.complex64)
