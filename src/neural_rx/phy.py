# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Narrow Sionna 2 adapter for PUSCH receiver training."""

from __future__ import annotations

import ast
import configparser
import math
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn

from sionna.phy import config as sionna_config
from sionna.phy.channel import AWGN, OFDMChannel
from sionna.phy.channel.tr38901 import PanelArray, TDL, UMa, UMi
from sionna.phy.mapping import Constellation, Mapper
from sionna.phy.nr import (
    CarrierConfig,
    PUSCHConfig,
    PUSCHDMRSConfig,
    PUSCHLSChannelEstimator,
    PUSCHPilotPattern,
    PUSCHTransmitter,
    TBConfig,
    TBDecoder,
    TBEncoder,
)
from sionna.phy.ofdm import OFDMDemodulator, OFDMModulator
from sionna.phy.utils import ebnodb2no

from neural_rx.config import load_experiment_config


class PuschBatch(NamedTuple):
    received: Tensor
    pilot_features: Tensor
    active_users: Tensor
    initial_channel: Tensor | None
    channel_target: Tensor
    bit_targets: Tensor
    data_mask: Tensor
    bit_mask: Tensor
    mcs_mask: Tensor
    information_bits: Tensor | tuple[Tensor, ...]


def sample_num_active_users(
    minimum: int, maximum: int, *, device: str | torch.device = "cpu"
) -> Tensor:
    """Sample a triangular distribution biased toward more active users."""

    if minimum < 1 or maximum < minimum:
        raise ValueError("Active-user limits must satisfy 1 <= minimum <= maximum")
    uniform = torch.rand((), device=device)
    sample = minimum + torch.sqrt(uniform) * (maximum + 1 - minimum)
    return sample.floor().to(torch.int64).clamp_max(maximum)


def _single_sector_topology(
    batch_size: int,
    num_users: int,
    min_speed: float,
    max_speed: float,
    device: torch.device,
    scenario: str = "UMi",
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Generate the outdoor UMi or UMa experiment topology."""

    dtype = torch.float32
    minimum_distance, inter_site_distance, bs_height = (
        (10.0, 200.0, 10.0) if scenario == "UMi" else (35.0, 500.0, 25.0)
    )
    ut_height = 1.5
    shape = (batch_size, num_users)

    alpha_half = torch.empty(shape, device=device, dtype=dtype).uniform_(-math.pi / 6, math.pi / 6)
    maximum_radius = (inter_site_distance / 2) / torch.cos(alpha_half)
    distance_squared = minimum_distance**2 + torch.rand(shape, device=device, dtype=dtype) * (
        maximum_radius.square() - minimum_distance**2
    )
    side = 2 * torch.bernoulli(torch.full(shape, 0.5, device=device)) + 1
    angle = alpha_half + side * math.pi / 6
    ut_xy = torch.stack(
        [distance_squared.sqrt() * torch.cos(angle), distance_squared.sqrt() * torch.sin(angle)],
        dim=-1,
    )
    ut_loc = torch.cat(
        [ut_xy, torch.full((*shape, 1), ut_height, device=device, dtype=dtype)], dim=-1
    )

    velocity_angle = torch.empty(shape, device=device, dtype=dtype).uniform_(-math.pi, math.pi)
    if min_speed == max_speed:
        velocity_norm = torch.full(shape, min_speed, device=device, dtype=dtype)
    else:
        velocity_norm = torch.empty(shape, device=device, dtype=dtype).uniform_(
            min_speed, max_speed
        )
    ut_velocities = torch.stack(
        [
            velocity_norm * torch.cos(velocity_angle),
            velocity_norm * torch.sin(velocity_angle),
            torch.zeros(shape, device=device, dtype=dtype),
        ],
        dim=-1,
    )
    ut_orientations = torch.empty((*shape, 3), device=device, dtype=dtype).uniform_(
        -math.pi / 2, math.pi / 2
    )
    bs_loc = torch.zeros((batch_size, 1, 3), device=device, dtype=dtype)
    bs_loc[..., 2] = bs_height
    sector_center = (minimum_distance + 0.5 * inter_site_distance) * 0.5
    bs_orientation = torch.zeros((batch_size, 1, 3), device=device, dtype=dtype)
    bs_orientation[..., 0] = math.pi / 3
    bs_orientation[..., 1] = math.pi / 2 - math.atan(sector_center / bs_height)
    in_state = torch.zeros(shape, device=device, dtype=torch.bool)
    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities, in_state


def _experiment_values(path: Path) -> dict[str, dict[str, Any]]:
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    if not parser.read(path, encoding="utf-8"):
        raise FileNotFoundError(path)
    values: dict[str, dict[str, Any]] = {}
    for section in parser.sections():
        values[section] = {}
        for option, raw in parser.items(section):
            try:
                values[section][option] = ast.literal_eval(raw)
            except (SyntaxError, ValueError) as exc:
                raise ValueError(f"Invalid value for [{section}] {option}: {raw!r}") from exc
    return values


def _correlation_matrix(num_antennas: int, alpha: float) -> Tensor:
    """Antenna correlation of the 3GPP MIMO test channels (TS 38.104, Annex G)."""

    exponents = torch.linspace(0, 1, num_antennas).square()
    index = torch.arange(num_antennas)
    distance = (index[:, None] - index[None]).abs()
    return (alpha ** exponents[distance]).to(torch.complex64)


class FrequencyOffset:
    """Apply a carrier frequency offset per user to a frequency-domain signal.

    The offset is relative to the sampling rate and drawn uniformly from
    [-max_rel_offset, max_rel_offset], or fixed to max_rel_offset if constant.
    """

    def __init__(self, max_rel_offset: float, resource_grid: Any, constant: bool, device: str):
        self.max_rel_offset = max_rel_offset
        self.constant = constant
        self.modulator = OFDMModulator(resource_grid.cyclic_prefix_length, device=device)
        self.demodulator = OFDMDemodulator(
            resource_grid.fft_size, 0, resource_grid.cyclic_prefix_length, device=device
        )

    def __call__(self, signal: Tensor) -> Tensor:
        time_signal = self.modulator(signal)
        offset = torch.full(
            (*signal.shape[:2], 1, 1), self.max_rel_offset, device=signal.device
        )
        if not self.constant:
            offset.uniform_(-self.max_rel_offset, self.max_rel_offset)
        steps = torch.arange(time_signal.shape[-1], device=signal.device)
        time_signal = time_signal * torch.polar(
            torch.ones_like(offset), 2 * math.pi * offset * steps
        )
        return self.demodulator(time_signal)


def _pilot_features(mask: Tensor) -> Tensor:
    users, symbols, subcarriers = mask.shape
    symbol_grid = torch.arange(symbols, device=mask.device)
    subcarrier_grid = torch.arange(subcarriers, device=mask.device)
    features = []
    for user in range(users):
        positions = torch.nonzero(mask[user], as_tuple=False)
        if positions.numel() == 0:
            raise ValueError("Every user must have at least one pilot resource element")
        symbol_distance = (symbol_grid[:, None] - positions[:, 0]).abs().amin(dim=1).float()
        carrier_distance = (subcarrier_grid[:, None] - positions[:, 1]).abs().amin(dim=1).float()
        symbol_distance = (symbol_distance - symbol_distance.mean()) / symbol_distance.std(
            correction=0
        ).clamp_min(1.0)
        carrier_distance = (carrier_distance - carrier_distance.mean()) / carrier_distance.std(
            correction=0
        ).clamp_min(1.0)
        time_feature = symbol_distance[:, None].expand(-1, subcarriers)
        frequency_feature = carrier_distance[None, :].expand(symbols, -1)
        features.append(torch.stack([time_feature, frequency_feature], dim=-1))
    return torch.stack(features).permute(0, 2, 1, 3).contiguous()


class SionnaPuschAdapter:
    """Generate PUSCH batches for the retained NRX and E2E experiments."""

    def __init__(
        self,
        pusch_configs: list[PUSCHConfig] | list[list[PUSCHConfig]],
        *,
        num_rx_antennas: int,
        carrier_frequency: float,
        normalize_channel: bool,
        channel_type: str,
        min_ut_velocity: float,
        max_ut_velocity: float,
        decoder_iterations: int,
        decoder_cn_type: str,
        initial_channel_estimate: bool,
        custom_constellation: bool,
        mask_pilots: bool,
        randomize_pilots: bool,
        rate_adjusted_snr: bool,
        dataset_path: str | Path | None,
        dataset_training: bool,
        random_subsampling: bool,
        max_dataset_samples: int | None,
        device: str,
        channel_chunk: int = 1,
        cfo_offset_ppm: float = 0.0,
        constant_cfo: bool = False,
    ) -> None:
        if channel_chunk < 1:
            raise ValueError("channel_chunk must be positive")
        self.channel_chunk = channel_chunk
        self._channel_cache: Tensor | None = None
        self.device = torch.device(device)
        self.carrier_frequency = carrier_frequency
        self.num_rx_antennas = num_rx_antennas
        sionna_config.device = device
        sionna_config.precision = "single"
        if pusch_configs and isinstance(pusch_configs[0], PUSCHConfig):
            config_sets = [pusch_configs]
        else:
            config_sets = pusch_configs
        self.pusch_config_sets = config_sets
        self.pusch_configs = config_sets[0]
        if not config_sets or any(
            len(configs) != len(self.pusch_configs) for configs in config_sets
        ):
            raise ValueError("Every MCS must define the same configured users")
        first = pusch_configs[0]
        if isinstance(first, list):
            first = first[0]
        self.num_users = len(self.pusch_configs)
        self.channel_type = channel_type
        self.min_ut_velocity = min_ut_velocity
        self.max_ut_velocity = max_ut_velocity
        self.decoder_iterations = decoder_iterations
        self.mask_pilots = mask_pilots
        self.randomize_pilots = randomize_pilots
        self.rate_adjusted_snr = rate_adjusted_snr
        slot_numbers = [config.carrier.slot_number for config in self.pusch_configs]
        pilot_set = []
        slots = first.carrier.num_slots_per_frame if randomize_pilots else 1
        for slot in range(slots):
            if randomize_pilots:
                for configs in config_sets:
                    for config in configs:
                        config.carrier.slot_number = slot
            pilot_set.append(PUSCHPilotPattern(self.pusch_configs).pilots)
        for configs in config_sets:
            for config, slot in zip(configs, slot_numbers):
                config.carrier.slot_number = slot
        self.pilot_set = torch.stack(pilot_set).to(self.device)
        self.transmitters = [
            PUSCHTransmitter(configs, return_bits=True, output_domain="freq", device=device)
            for configs in config_sets
        ]
        self.transmitter = self.transmitters[0]
        # Sionna 2.1 does not expose a public mapper replacement API.
        # Keep this access localized; the exact Sionna version is pinned.
        self.mappers = [transmitter._mapper for transmitter in self.transmitters]
        self.mapper = self.mappers[0]
        self.resource_grid = self.transmitter.resource_grid
        self.num_bits_by_mcs = tuple(
            int(configs[0].tb.num_bits_per_symbol.item()) for configs in config_sets
        )
        self.coderates = tuple(
            float(configs[0].tb.target_coderate.item()) for configs in config_sets
        )
        self.num_bits = max(self.num_bits_by_mcs)
        self.coderate = self.coderates[0]
        if custom_constellation:
            self.mappers = []
            for transmitter, num_bits in zip(self.transmitters, self.num_bits_by_mcs):
                qam = Constellation("qam", num_bits, device=device)
                constellation = Constellation(
                    "custom",
                    num_bits,
                    points=nn.Parameter(qam().detach().clone()),
                    normalize=True,
                    center=True,
                    device=device,
                )
                mapper = Mapper(constellation=constellation, device=device)
                transmitter._mapper = mapper
                self.mappers.append(mapper)
            self.mapper = self.mappers[0]
        self.encoders = [
            TBEncoder(
                target_tb_size=configs[0].tb_size,
                num_coded_bits=configs[0].num_coded_bits,
                target_coderate=coderate,
                num_bits_per_symbol=num_bits,
                num_layers=configs[0].num_layers,
                n_rnti=[config.n_rnti for config in configs],
                n_id=[
                    config.tb.n_id if config.tb.n_id is not None else config.carrier.n_cell_id
                    for config in configs
                ],
                channel_type="PUSCH",
                device=device,
            )
            for configs, coderate, num_bits in zip(
                config_sets, self.coderates, self.num_bits_by_mcs
            )
        ]
        self.encoder = self.encoders[0]
        cn_update = {"boxplus": "boxplus-phi"}.get(decoder_cn_type, decoder_cn_type)
        self.decoders = [
            TBDecoder(encoder, num_bp_iter=decoder_iterations, cn_update=cn_update, device=device)
            for encoder in self.encoders
        ]
        self.decoder = self.decoders[0]
        self.channel_model: Any | None = None
        self.channel: OFDMChannel | None = None
        self.channels: list[OFDMChannel] = []
        if channel_type == "Dataset":
            if dataset_path is None:
                raise ValueError("Dataset channels require a CIR dataset path")
            from neural_rx.cir_dataset import CIRDatasetChannel

            self.channel_model = CIRDatasetChannel(
                dataset_path,
                training=dataset_training,
                num_tx=self.num_users,
                random_subsampling=random_subsampling,
                max_num_samples=max_dataset_samples,
                device=device,
            )
            self.channel = OFDMChannel(
                self.channel_model,
                self.resource_grid,
                normalize_channel=normalize_channel,
                return_channel=True,
                device=device,
            )
        elif channel_type in {"UMi", "UMa"}:
            dual_polarized = num_rx_antennas % 2 == 0
            bs_array = PanelArray(
                num_rows_per_panel=1,
                num_cols_per_panel=(
                    num_rx_antennas // 2 if dual_polarized else num_rx_antennas
                ),
                polarization="dual" if dual_polarized else "single",
                polarization_type="cross" if dual_polarized else "V",
                antenna_pattern="38.901",
                carrier_frequency=carrier_frequency,
                device=device,
            )
            ut_array = PanelArray(
                num_rows_per_panel=1,
                num_cols_per_panel=first.num_antenna_ports,
                polarization="single",
                polarization_type="V",
                antenna_pattern="omni",
                carrier_frequency=carrier_frequency,
                device=device,
            )
            self.channel_model = (UMi if channel_type == "UMi" else UMa)(
                carrier_frequency=carrier_frequency,
                o2i_model="low",
                ut_array=ut_array,
                bs_array=bs_array,
                direction="uplink",
                enable_pathloss=False,
                enable_shadow_fading=False,
                device=device,
            )
            self.channel = OFDMChannel(
                self.channel_model,
                self.resource_grid,
                normalize_channel=normalize_channel,
                return_channel=True,
                device=device,
            )
        elif channel_type != "AWGN":
            correlation = {"DoubleTDLlow": (0.0, 0.0), "DoubleTDLmedium": (0.9, 0.3),
                           "DoubleTDLhigh": (0.9, 0.9)}.get(channel_type)
            if correlation is not None:
                speed_of_light = 299_792_458.0
                channel_specs = (
                    ("B100", 100e-9, 400.0 * speed_of_light / carrier_frequency),
                    ("C300", 300e-9, 100.0 * speed_of_light / carrier_frequency),
                )
            elif channel_type in {"TDL-B100", "TDL-C300"}:
                model = channel_type.removeprefix("TDL-")
                delay_spread = 100e-9 if model == "B100" else 300e-9
                channel_specs = ((model, delay_spread, None),)
            else:
                raise ValueError(f"Unsupported channel type: {channel_type!r}")
            if len(channel_specs) != self.num_users:
                raise ValueError(
                    f"{channel_type} requires {len(channel_specs)} user(s), got {self.num_users}"
                )
            for config, (model, delay_spread, maximum_doppler_speed) in zip(
                self.pusch_configs, channel_specs
            ):
                minimum_speed = min_ut_velocity if maximum_doppler_speed is None else 0.0
                maximum_speed = (
                    max_ut_velocity if maximum_doppler_speed is None else maximum_doppler_speed
                )
                rx_correlation = None
                tx_correlation = None
                if correlation is not None:
                    rx_correlation = _correlation_matrix(num_rx_antennas, correlation[0])
                    tx_correlation = _correlation_matrix(config.num_antenna_ports, correlation[1])
                channel_model = TDL(
                    model=model,
                    delay_spread=delay_spread,
                    carrier_frequency=carrier_frequency,
                    min_speed=minimum_speed,
                    max_speed=maximum_speed,
                    num_tx_ant=config.num_antenna_ports,
                    num_rx_ant=num_rx_antennas,
                    rx_corr_mat=rx_correlation,
                    tx_corr_mat=tx_correlation,
                    device=device,
                )
                self.channels.append(
                    OFDMChannel(
                        channel_model,
                        self.resource_grid,
                        normalize_channel=normalize_channel,
                        return_channel=True,
                        device=device,
                    )
                )
        self.awgn = AWGN(device=device)
        self.frequency_offset = None
        if cfo_offset_ppm > 0:
            offset = carrier_frequency / 1e6 * cfo_offset_ppm
            self.frequency_offset = FrequencyOffset(
                offset / self.resource_grid.bandwidth, self.resource_grid, constant_cfo, device
            )
        self.channel_estimator = None
        if initial_channel_estimate:
            self.channel_estimator = PUSCHLSChannelEstimator(
                self.resource_grid,
                dmrs_length=first.dmrs.length,
                dmrs_additional_position=first.dmrs.additional_position,
                num_cdm_groups_without_data=first.dmrs.num_cdm_groups_without_data,
                interpolation_type="nn",
                device=device,
            )
        resource_types = self.resource_grid.build_type_grid()[:, 0]
        self.data_mask = (resource_types == 0).permute(0, 2, 1).contiguous()
        pilot_mask = self.transmitter.pilot_pattern.mask[:, 0].bool()
        self.pilot_mask = pilot_mask.permute(0, 2, 1).contiguous()
        self.pilot_features = _pilot_features(pilot_mask)

    @classmethod
    def from_config(
        cls,
        path: str | Path,
        *,
        mode: str = "training",
        training_channel_type: str | None = None,
        device: str = "cpu",
        dataset_path_override: str | Path | None = None,
        num_users_override: int | None = None,
        n_size_bwp_override: int | None = None,
    ) -> "SionnaPuschAdapter":
        if mode not in {"training", "evaluation"}:
            raise ValueError("mode must be 'training' or 'evaluation'")
        values = _experiment_values(Path(path))
        experiment = load_experiment_config(path)
        system = values["system"]
        mcs_indices = tuple(system["mcs_index"])

        if mode == "training":
            channel_type = (
                experiment.training.channel_type
                if training_channel_type is None
                else training_channel_type
            )
            normalize_channel = experiment.training.normalize_channel
            min_ut_velocity = experiment.training.min_ut_velocity
            max_ut_velocity = experiment.training.max_ut_velocity
            n_size_bwp = system["n_size_bwp"]
            num_users = experiment.max_num_tx
            dataset_path = experiment.training.dataset_path
            random_subsampling = experiment.training.random_subsampling
            max_dataset_samples = experiment.training.max_dataset_samples
            cfo_offset_ppm = experiment.training.cfo_offset_ppm
        else:
            if training_channel_type is not None:
                raise ValueError("training_channel_type is only valid in training mode")
            channel_type = experiment.evaluation.channel_type
            normalize_channel = experiment.evaluation.normalize_channel
            min_ut_velocity = experiment.evaluation.min_ut_velocity
            max_ut_velocity = experiment.evaluation.max_ut_velocity
            n_size_bwp = experiment.evaluation.n_size_bwp
            num_users = (
                experiment.max_num_tx
                if channel_type == "Dataset"
                else 2 if channel_type.startswith("DoubleTDL") else 1
            )
            cfo_offset_ppm = experiment.evaluation.cfo_offset_ppm
            dataset_path = experiment.evaluation.dataset_path
            random_subsampling = experiment.evaluation.random_subsampling
            max_dataset_samples = experiment.evaluation.max_dataset_samples
        if num_users_override is not None:
            if not 1 <= num_users_override <= experiment.max_num_tx:
                raise ValueError("num_users_override exceeds configured user limits")
            num_users = num_users_override
            # One-user evaluation uses TDL-B100 rather than muting one
            # link of the two-user DoubleTDL channel.
            if num_users == 1 and channel_type.startswith("DoubleTDL"):
                channel_type = "TDL-B100"
        if n_size_bwp_override is not None:
            if n_size_bwp_override < 1:
                raise ValueError("n_size_bwp_override must be positive")
            n_size_bwp = n_size_bwp_override
        if dataset_path_override is not None:
            dataset_path = Path(dataset_path_override).resolve()
        if dataset_path is not None:
            dataset_path = Path(dataset_path)
            if not dataset_path.is_absolute():
                dataset_path = Path(path).resolve().parent / dataset_path

        user_fields = ("dmrs_port_sets", "dmrs_nid", "n_rntis", "n_ids")
        for field in user_fields:
            if len(system[field]) < num_users:
                raise ValueError(f"[system] {field} must define all {num_users} configured users")
        sionna_config.device = device
        carrier = CarrierConfig(
            n_cell_id=system["n_cell_id"],
            cyclic_prefix=system["cyclic_prefix"],
            subcarrier_spacing=int(system["subcarrier_spacing"] / 1e3),
            n_size_grid=n_size_bwp,
            n_start_grid=system["n_start_grid"],
            slot_number=system["slot_number"],
            frame_number=system["frame_number"],
        )
        dmrs = PUSCHDMRSConfig(
            config_type=system["dmrs_config_type"],
            type_a_position=system["dmrs_type_a_position"],
            additional_position=system["dmrs_additional_position"],
            length=system["dmrs_length"],
            dmrs_port_set=system["dmrs_port_sets"][0],
            n_scid=system["n_scid"],
            num_cdm_groups_without_data=system["num_cdm_groups_without_data"],
        )
        pusch_configs = []
        for mcs_index in mcs_indices:
            tb = TBConfig(
                mcs_index=mcs_index,
                mcs_table=system["mcs_table"],
                channel_type="PUSCH",
            )
            template = PUSCHConfig(
                carrier_config=carrier,
                pusch_dmrs_config=dmrs,
                tb_config=tb,
                num_antenna_ports=system["num_antenna_ports"],
                precoding=system["precoding"],
                symbol_allocation=system["symbol_allocation"],
                tpmi=system["tpmi"],
                mapping_type=system["dmrs_mapping_type"],
            )
            configs = []
            for user in range(num_users):
                config = template.clone()
                config.dmrs.dmrs_port_set = system["dmrs_port_sets"][user]
                config.dmrs.n_id = system["dmrs_nid"][user]
                config.n_rnti = system["n_rntis"][user]
                config.tb.n_id = system["n_ids"][user]
                configs.append(config)
            pusch_configs.append(configs)
        return cls(
            pusch_configs,
            num_rx_antennas=system["num_rx_antennas"],
            carrier_frequency=system["carrier_frequency"],
            normalize_channel=normalize_channel,
            channel_type=channel_type,
            min_ut_velocity=min_ut_velocity,
            max_ut_velocity=max_ut_velocity,
            decoder_iterations=experiment.decoder_iterations,
            decoder_cn_type=experiment.decoder_cn_type,
            initial_channel_estimate=experiment.receiver.initial_channel_estimate,
            custom_constellation=experiment.custom_constellation,
            mask_pilots=experiment.mask_pilots,
            randomize_pilots=(experiment.randomize_pilots and mode == "training"),
            rate_adjusted_snr=experiment.rate_adjusted_snr,
            dataset_path=dataset_path,
            dataset_training=mode == "training",
            random_subsampling=random_subsampling,
            max_dataset_samples=max_dataset_samples,
            device=device,
            channel_chunk=32 if mode == "training" else 1,
            cfo_offset_ppm=cfo_offset_ppm,
            constant_cfo=mode == "evaluation",
        )

    def _umi_channel(self, batch_size: int) -> Tensor:
        """Return UMi/UMa frequency responses, generating several batches per topology draw."""

        if self._channel_cache is None or self._channel_cache.shape[0] < batch_size:
            num_samples = batch_size * self.channel_chunk
            self.channel_model.set_topology(*_single_sector_topology(
                num_samples, self.num_users, self.min_ut_velocity, self.max_ut_velocity,
                self.device, self.channel_type,
            ))
            self._channel_cache = self.channel.generate(num_samples)
        channel_frequency = self._channel_cache[:batch_size]
        self._channel_cache = self._channel_cache[batch_size:]
        return channel_frequency

    def _bit_target_grid(self, coded_bits: Tensor, num_bits: int) -> Tensor:
        batch, users, _ = coded_bits.shape
        symbols = self.resource_grid.num_ofdm_symbols
        subcarriers = self.resource_grid.fft_size
        targets = []
        for user in range(users):
            user_bits = coded_bits[:, user].reshape(batch, -1, num_bits)
            grid = coded_bits.new_zeros((batch, symbols, subcarriers, self.num_bits))
            user_bits = torch.nn.functional.pad(user_bits, (0, self.num_bits - num_bits))
            grid[:, self.data_mask[user].transpose(0, 1)] = user_bits
            targets.append(grid.permute(0, 2, 1, 3))
        return torch.stack(targets, dim=1)

    def codeword_llrs(self, llr_grid: Tensor, mcs_index: int = 0) -> Tensor:
        """Return full-grid LLRs in PUSCH codeword order."""

        codewords = []
        num_bits = self.num_bits_by_mcs[mcs_index]
        for user in range(llr_grid.shape[1]):
            grid = llr_grid[:, user, ..., :num_bits].permute(0, 2, 1, 3)
            codewords.append(grid[:, self.data_mask[user].transpose(0, 1)].flatten(1))
        return torch.stack(codewords, dim=1)

    def compile_decoders(self) -> None:
        """Compile the LDPC BP iteration on CUDA; it dominates evaluation time."""

        if self.device.type == "cuda":
            for decoder in self.decoders:
                bp = decoder._decoder
                bp._bp_iter = torch.compile(bp._bp_iter, fullgraph=True)

    def decode(self, llr_grid: Tensor, mcs_index: int = 0) -> tuple[Tensor, Tensor]:
        """Decode one selected MCS and return information bits and CRC status."""

        return self.decoders[mcs_index](self.codeword_llrs(llr_grid, mcs_index))

    @property
    def transmitter_parameters(self) -> tuple[nn.Parameter, ...]:
        """Return trainable transmitter parameters, if configured."""

        return tuple(parameter for mapper in self.mappers for parameter in mapper.parameters())

    @property
    def checkpoint_tensors(self) -> dict[str, Tensor]:
        """Return transmitter tensors that must accompany receiver weights."""

        parameters = self.transmitter_parameters
        if not parameters:
            return {}
        if len(parameters) == 1:
            return {"constellation_points": parameters[0]}
        return {f"constellation_points_{index}": value for index, value in enumerate(parameters)}

    @property
    def constellation_points(self) -> Tensor:
        """Return centered, normalized points used by the transmitter."""

        return self.mapper.constellation()

    def set_transmitter_trainable(self, trainable: bool) -> None:
        """Enable or freeze end-to-end transmitter optimization."""

        for parameter in self.transmitter_parameters:
            parameter.requires_grad_(trainable)

    def _effective_channel(
        self, channel_frequency: Tensor, channel_user: int, config: PUSCHConfig
    ) -> Tensor:
        channel_frequency = channel_frequency[:, 0, :, channel_user]
        if config.precoding_matrix is None:
            precoder = torch.eye(
                channel_frequency.shape[2],
                config.num_layers,
                dtype=channel_frequency.dtype,
                device=self.device,
            )
        else:
            precoder = torch.as_tensor(
                config.precoding_matrix,
                dtype=channel_frequency.dtype,
                device=self.device,
            )
        effective = torch.einsum("bratf,al->brltf", channel_frequency, precoder)
        effective = effective[:, :, 0].permute(0, 3, 2, 1)
        return torch.cat([effective.real, effective.imag], dim=-1)

    def sample(
        self,
        batch_size: int,
        ebno_db: float | Tensor,
        *,
        num_active_users: int | None = None,
        active_users: Tensor | None = None,
        mcs_mask: Tensor | None = None,
        snr_mcs_index: int = 0,
    ) -> PuschBatch:
        """Sample one batch; ``active_users`` overrides activating the first users."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if num_active_users is None:
            num_active_users = self.num_users
        if not 1 <= num_active_users <= self.num_users:
            raise ValueError("num_active_users is outside the configured user range")

        if self.randomize_pilots:
            index = torch.randint(len(self.pilot_set), (), device=self.device)
            for transmitter in self.transmitters:
                transmitter.pilot_pattern.pilots = self.pilot_set[index]
        if mcs_mask is None:
            if len(self.transmitters) == 1:
                mcs_indices = torch.zeros(
                    (batch_size, self.num_users), dtype=torch.int64, device=self.device
                )
            else:
                mcs_indices = torch.randint(
                    len(self.transmitters),
                    (batch_size, self.num_users),
                    device=self.device,
                )
            mcs_mask = torch.nn.functional.one_hot(
                mcs_indices, num_classes=len(self.transmitters)
            ).to(torch.float32)
        else:
            mcs_mask = mcs_mask.to(device=self.device, dtype=torch.float32)
            if mcs_mask.shape != (batch_size, self.num_users, len(self.transmitters)):
                raise ValueError("MCS mask must have shape [B,U,M]")
        signals = []
        information_bits_by_mcs = []
        for transmitter in self.transmitters:
            signal, bits = transmitter(batch_size)
            signals.append(signal)
            information_bits_by_mcs.append(bits)
        transmitted = sum(
            signal * mcs_mask[:, :, index, None, None, None]
            for index, signal in enumerate(signals)
        )
        if active_users is None:
            active_users = transmitted.real.new_zeros((batch_size, self.num_users))
            active_users[:, :num_active_users] = 1
        elif active_users.shape != (batch_size, self.num_users):
            raise ValueError("Active users must have shape [B,U]")
        else:
            active_users = active_users.to(transmitted.real)
        transmitted = transmitted * active_users[:, :, None, None, None]
        if self.frequency_offset is not None:  # per user, i.e., at the transmitter
            transmitted = self.frequency_offset(transmitted)

        channel_targets = []
        if self.channel_type == "AWGN":
            received = transmitted.sum(dim=(1, 2))[:, None, None].expand(
                -1, 1, self.num_rx_antennas, -1, -1
            )
            ones = torch.ones_like(transmitted)[:, None, None].expand(
                -1, 1, self.num_rx_antennas, -1, -1, -1, -1
            )
            for user, config in enumerate(self.pusch_configs):
                channel_targets.append(self._effective_channel(ones, user, config))
        elif self.channel_model is not None:
            if self.channel is None:
                raise RuntimeError("Shared channel was not initialized")
            if self.channel_type in {"UMi", "UMa"}:
                channel_frequency = self._umi_channel(batch_size)
                received = self.channel.apply(transmitted, channel_frequency)
            else:
                received, channel_frequency = self.channel(transmitted)
            for user, config in enumerate(self.pusch_configs):
                channel_targets.append(self._effective_channel(channel_frequency, user, config))
        else:
            user_signals = []
            for user, (channel, config) in enumerate(zip(self.channels, self.pusch_configs)):
                user_received, channel_frequency = channel(transmitted[:, user : user + 1])
                user_signals.append(user_received)
                channel_targets.append(self._effective_channel(channel_frequency, 0, config))
            received = torch.stack(user_signals).sum(dim=0)

        noise_variance = self.noise_variance(ebno_db, mcs_index=snr_mcs_index)
        received = self.awgn(received, noise_variance)
        initial_channel = None
        if self.channel_estimator is not None:
            initial_channel, _ = self.channel_estimator(received, noise_variance)
            initial_channel = initial_channel[:, 0, :, :, 0].permute(0, 2, 4, 3, 1)
            initial_channel = torch.cat([initial_channel.real, initial_channel.imag], dim=-1)
        received = received[:, 0].permute(0, 3, 2, 1)
        received = torch.cat([received.real, received.imag], dim=-1)
        if self.mask_pilots:
            mask = self.pilot_mask.any(dim=0)[None, :, :, None]
            received = torch.where(mask, torch.zeros_like(received), received)
        channel_target = torch.stack(channel_targets, dim=1)
        bit_targets_by_mcs = [
            self._bit_target_grid(encoder(bits), num_bits)
            for encoder, bits, num_bits in zip(
                self.encoders, information_bits_by_mcs, self.num_bits_by_mcs
            )
        ]
        bit_targets = sum(
            targets * mcs_mask[:, :, index, None, None, None]
            for index, targets in enumerate(bit_targets_by_mcs)
        )
        bit_indices = torch.arange(self.num_bits, device=self.device)
        bits_per_mcs = torch.tensor(self.num_bits_by_mcs, device=self.device)
        bit_mask = torch.matmul(
            mcs_mask, (bit_indices[None] < bits_per_mcs[:, None]).to(mcs_mask.dtype)
        )
        information_bits: Tensor | tuple[Tensor, ...]
        information_bits = (
            information_bits_by_mcs[0]
            if len(information_bits_by_mcs) == 1
            else tuple(information_bits_by_mcs)
        )
        return PuschBatch(
            received=received,
            pilot_features=self.pilot_features,
            active_users=active_users,
            initial_channel=initial_channel,
            channel_target=channel_target,
            bit_targets=bit_targets,
            data_mask=self.data_mask,
            bit_mask=bit_mask,
            mcs_mask=mcs_mask,
            information_bits=information_bits,
        )

    def noise_variance(self, ebno_db: float | Tensor, *, mcs_index: int = 0) -> Tensor:
        """Return noise power referenced to the evaluated MCS.

        Training retains the first-MCS convention unless explicitly overridden.
        For mixed-MCS evaluation, choose the MCS of the reported user.
        """
        if not 0 <= mcs_index < len(self.num_bits_by_mcs):
            raise ValueError("Noise-reference MCS index is outside the configured range")
        snr = torch.as_tensor(ebno_db, dtype=torch.float32, device=self.device)
        if not self.rate_adjusted_snr:
            return torch.pow(10.0, -snr / 10.0)
        if self.mask_pilots:
            pilot_fraction = float(self.resource_grid.num_pilot_symbols) / float(
                self.resource_grid.num_resource_elements
            )
            snr = snr - 10 * math.log10(1 - pilot_fraction)
        return ebnodb2no(snr, self.num_bits_by_mcs[mcs_index],
                        self.coderates[mcs_index], self.resource_grid,
                        device=str(self.device))
