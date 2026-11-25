# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

"""Classical receiver baselines implemented with Sionna PHY algorithms."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor
from sionna.phy.channel import GenerateOFDMChannel, gen_single_sector_topology
from sionna.phy.mimo import StreamManagement
from sionna.phy.nr import LayerDemapper, PUSCHLSChannelEstimator, PUSCHLMMSEChannelEstimator, PUSCHTransmitter
from sionna.phy.ofdm import KBestDetector, LinearDetector

from neural_rx.phy import PuschBatch, SionnaPuschAdapter


@dataclass(frozen=True)
class ChannelCovariance:
    time: Tensor
    frequency: Tensor
    space: Tensor


def covariance_from_channels(h: Tensor, frequency_size: int | None = None) -> ChannelCovariance:
    """Uncentered covariance of normalized raw channels [B,R,T,F].

    Only the frequency covariance may be cropped; time and space use the
    entire normalized bandwidth.
    Contractions avoid allocating [B,R,F,F] intermediate tensors.
    """
    if h.ndim != 4 or not h.is_complex():
        raise ValueError('Channels must be complex [B,R,T,F] tensors')
    b, r, t, f = h.shape
    hf = h if frequency_size is None else h[..., :frequency_size]
    return ChannelCovariance(
        torch.einsum('brtf,brsf->ts', h, h.conj()) / (b*r*f),
        torch.einsum('brtf,brtg->fg', hf, hf.conj()) / (b*r*t),
        torch.einsum('brtf,bstf->rs', h, h.conj()) / (b*t*f),
    )


@torch.no_grad()
def estimate_covariance(adapter: SionnaPuschAdapter, num_samples: int,
                        batch_size: int = 1, frequency_size: int | None = None) -> ChannelCovariance:
    """Estimate normalized UMi first-user/first-port covariances.

    Pass a UMi adapter even for TDL or Dataset evaluations. Sampling uses
    floor(num_samples/batch_size)+1 complete batches and retains UMi's
    default 80% indoor population.
    """
    if adapter.channel_type != 'UMi' or adapter.channel_model is None:
        raise ValueError('Baseline covariance requires a UMi adapter, including for Dataset/TDL evaluation')
    if num_samples < 1 or batch_size < 1:
        raise ValueError('Sample and batch counts must be positive')
    generator = GenerateOFDMChannel(adapter.channel_model, adapter.resource_grid,
                                    normalize_channel=True, device=str(adapter.device))
    iterations = num_samples // batch_size + 1
    total = None
    for _ in range(iterations):
        # Covariance estimation samples one UT regardless of detector stream count.
        topology = gen_single_sector_topology(
            batch_size, 1, 'umi', min_ut_velocity=adapter.min_ut_velocity,
            max_ut_velocity=adapter.max_ut_velocity, device=str(adapter.device))
        adapter.channel_model.set_topology(*topology)
        h = generator(batch_size)[:, 0, :, 0, 0]
        cov = covariance_from_channels(h, frequency_size)
        if total is None:
            total = [cov.time, cov.frequency, cov.space]
        else:
            for accumulator, value in zip(total, (cov.time, cov.frequency, cov.space)):
                accumulator.add_(value)
    return ChannelCovariance(*(value / iterations for value in total))


def lmmse_window_prbs(num_prbs: int) -> int:
    """Select the smallest divisor strictly above 20 PRBs."""
    if num_prbs < 1:
        raise ValueError('PRB count must be positive')
    if num_prbs <= 100:
        return num_prbs
    return next(n for n in range(21, num_prbs + 1) if num_prbs % n == 0)


def _interpolate(estimator, h: Tensor, error: Tensor) -> tuple[Tensor, Tensor]:
    """Native Sionna 2 scatter/filter stages, accepting already despread LS.

    Private state is localized here because Sionna 2 removed the standalone
    interpolator API. The filters themselves are unmodified native blocks.
    """
    shape = (*h.shape[:-1], estimator._num_ofdm_symbols, estimator._num_effective_subcarriers)
    h_grid, e_grid = h.new_zeros(shape), error.new_zeros(shape)
    tx, st, p, sb, sc = (getattr(estimator, '_scatter_' + name) for name in ('tx', 'st', 'p', 'sb', 'sc'))
    h_grid[:, :, :, tx, st, sb, sc] = h[:, :, :, tx, st, p]
    e_grid[:, :, :, tx, st, sb, sc] = error[:, :, :, tx, st, p]
    for order, filt, mask in zip(estimator._order, estimator._filters, estimator._err_var_masks):
        perm = {'f': (0, 1, 2, 3, 4, 5, 6), 't': (0, 1, 2, 3, 4, 6, 5),
                's': (0, 1, 3, 4, 5, 6, 2)}[order]
        inverse = tuple(perm.index(i) for i in range(7))
        h_grid, e_grid = filt(h_grid.permute(perm), e_grid.permute(perm))
        h_grid, e_grid = h_grid.permute(inverse), e_grid.permute(inverse)
        e_grid = e_grid * mask
    return h_grid, e_grid.clamp_min(0)


class _WindowedLMMSE:
    def __init__(self, adapter, covariance, window_prbs):
        configs = [pc.clone() for pc in adapter.pusch_configs]
        for pc in configs:
            pc.carrier.n_size_grid = window_prbs
        small = PUSCHTransmitter(configs, device=str(adapter.device)).resource_grid
        pc = configs[0]
        kwargs = dict(dmrs_length=pc.dmrs.length,
                      dmrs_additional_position=pc.dmrs.additional_position,
                      num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data,
                      device=str(adapter.device))
        self.ls = PUSCHLSChannelEstimator(adapter.resource_grid, **kwargs)
        self.native = PUSCHLMMSEChannelEstimator(small, **kwargs,
            cov_mat_time=covariance.time, cov_mat_freq=covariance.frequency[:small.fft_size, :small.fft_size],
            cov_mat_space=covariance.space, order='s-f-t')
        self.reduction = adapter.resource_grid.fft_size // small.fft_size
        self.symbols = pc.dmrs.length * (pc.dmrs.additional_position + 1)

    def __call__(self, y, no):
        h, error = self.ls.estimate_at_pilot_locations(self.ls._extract_pilots(y), no)
        error = error.expand_as(h)
        # Apply CDM despreading on the full bandwidth before reshaping pilot
        # symbols into contiguous frequency windows.
        prefix = h.shape[:-1]
        h = h.reshape(*prefix, self.symbols, self.reduction, -1)
        error = error.reshape_as(h)
        estimates, variances = [], []
        # Sequential windows bound memory while preserving independent solves.
        for window in range(self.reduction):
            hw = h[..., window, :].flatten(-2)
            ew = error[..., window, :].flatten(-2)
            result, variance = _interpolate(self.native, hw, ew)
            estimates.append(result)
            variances.append(variance)
        return torch.cat(estimates, -1), torch.cat(variances, -1)


class ClassicalReceiver:
    """LS-linear/LMMSE, sequential LMMSE/K-best64, or perfect-CSI/K-best64."""

    def __init__(self, adapter: SionnaPuschAdapter, kind: str,
                 covariance: ChannelCovariance | None = None, *, mcs_index: int = 0):
        if kind not in {'ls_linear_lmmse', 'lmmse_kbest', 'perfect_kbest'}:
            raise ValueError(f'Unknown classical baseline {kind!r}')
        if any(pc.num_layers != 1 for pc in adapter.pusch_configs):
            raise ValueError('Classical baselines require one layer per user')
        self.mcs_index = mcs_index
        self.adapter, self.kind = adapter, kind
        device = str(adapter.device)
        grid = adapter.resource_grid
        pc = adapter.pusch_configs[0]
        kwargs = dict(dmrs_length=pc.dmrs.length, dmrs_additional_position=pc.dmrs.additional_position,
                      num_cdm_groups_without_data=pc.dmrs.num_cdm_groups_without_data, device=device)
        self.estimator = None
        if kind == 'ls_linear_lmmse':
            self.estimator = PUSCHLSChannelEstimator(grid, interpolation_type='lin', **kwargs)
        elif kind == 'lmmse_kbest':
            if covariance is None:
                raise ValueError('LMMSE/K-best requires measured channel covariance')
            covariance = ChannelCovariance(*(v.to(adapter.device, torch.complex64)
                for v in (covariance.time, covariance.frequency, covariance.space)))
            prbs = grid.fft_size // 12
            window = lmmse_window_prbs(prbs)
            if window != prbs:
                self.estimator = _WindowedLMMSE(adapter, covariance, window)
            else:
                self.estimator = PUSCHLMMSEChannelEstimator(grid, **kwargs,
                    cov_mat_time=covariance.time, cov_mat_freq=covariance.frequency,
                    cov_mat_space=covariance.space, order='s-f-t')
        management = StreamManagement(np.ones((1, adapter.num_users), dtype=int), 1)
        num_bits = adapter.num_bits_by_mcs[mcs_index]
        common = dict(constellation_type='qam', num_bits_per_symbol=num_bits, device=device)
        if kind == 'ls_linear_lmmse':
            self.detector = LinearDetector('lmmse', 'bit', 'maxlog', grid, management, **common)
        else:
            self.detector = KBestDetector('bit', adapter.num_users, 64, grid, management, **common)
        self.demapper = LayerDemapper(adapter.transmitters[mcs_index]._layer_mapper,
                                      num_bits_per_symbol=num_bits, device=device)

    @torch.no_grad()
    def __call__(self, batch: PuschBatch, noise_variance) -> tuple[Tensor, Tensor]:
        r = self.adapter.num_rx_antennas
        y = torch.complex(batch.received[..., :r], batch.received[..., r:]).permute(0, 3, 2, 1).unsqueeze(1)
        no = torch.as_tensor(noise_variance, dtype=y.real.dtype, device=y.device)
        if self.estimator is None:
            h = torch.complex(batch.channel_target[..., :r], batch.channel_target[..., r:])
            h = h.permute(0, 4, 1, 3, 2).unsqueeze(1).unsqueeze(4)
            error = torch.zeros((), device=y.device)
        else:
            h, error = self.estimator(y, no)
        llr = self.detector(y, h, error, no)
        return self.adapter.decoders[self.mcs_index](self.demapper(llr))
