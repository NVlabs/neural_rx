# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary

from pathlib import Path

import pytest
import torch

from neural_rx.phy import SionnaPuschAdapter


def test_covariance_matches_explicit_outer_products():
    from neural_rx.classical import covariance_from_channels

    torch.manual_seed(1)
    h = torch.randn(2, 3, 4, 5, dtype=torch.complex64)
    cov = covariance_from_channels(h)
    expected_f = sum(torch.outer(h[b, r, t], h[b, r, t].conj())
                     for b in range(2) for r in range(3) for t in range(4)) / 24
    expected_t = sum(torch.outer(h[b, r, :, f], h[b, r, :, f].conj())
                     for b in range(2) for r in range(3) for f in range(5)) / 30
    expected_s = sum(torch.outer(h[b, :, t, f], h[b, :, t, f].conj())
                     for b in range(2) for t in range(4) for f in range(5)) / 40
    torch.testing.assert_close(cov.frequency, expected_f)
    torch.testing.assert_close(cov.time, expected_t)
    torch.testing.assert_close(cov.space, expected_s)


@pytest.mark.parametrize('kind', ['perfect_kbest', 'ls_linear_lmmse', 'lmmse_kbest'])
def test_classical_receivers_decode_small_high_snr_batch(kind):
    from neural_rx.classical import ClassicalReceiver, ChannelCovariance

    torch.manual_seed(3)
    adapter = SionnaPuschAdapter.from_config(
        Path(__file__).parents[1] / 'config/nrx_rt.cfg', device='cpu')
    batch = adapter.sample(1, 50.0)
    grid = adapter.resource_grid
    cov = ChannelCovariance(
        torch.ones(grid.num_ofdm_symbols, grid.num_ofdm_symbols, dtype=torch.complex64),
        torch.eye(grid.fft_size, dtype=torch.complex64),
        torch.eye(adapter.num_rx_antennas, dtype=torch.complex64))
    receiver = ClassicalReceiver(adapter, kind, covariance=cov)
    bits, crc = receiver(batch, 1e-5)
    assert bits.shape == batch.information_bits.shape
    assert crc.shape == batch.active_users.shape
    if kind == 'perfect_kbest':
        assert torch.equal(bits, batch.information_bits)
        assert crc.all()


def test_273_prb_split_is_21_prbs():
    from neural_rx.classical import lmmse_window_prbs

    assert lmmse_window_prbs(273) == 21
    assert lmmse_window_prbs(132) == 22
    assert lmmse_window_prbs(4) == 4


@pytest.mark.parametrize('window_prbs', [2, 4])
def test_windowed_lmmse_matches_native_for_uncorrelated_frequency(window_prbs):
    from neural_rx.classical import _WindowedLMMSE, ChannelCovariance
    from sionna.phy.nr import PUSCHLMMSEChannelEstimator

    adapter = SionnaPuschAdapter.from_config(
        Path(__file__).parents[1] / 'config/nrx_rt.cfg', device='cpu')
    grid = adapter.resource_grid
    cov = ChannelCovariance(
        torch.eye(grid.num_ofdm_symbols, dtype=torch.complex64),
        torch.eye(grid.fft_size, dtype=torch.complex64),
        torch.eye(adapter.num_rx_antennas, dtype=torch.complex64))
    pc = adapter.pusch_configs[0]
    native = PUSCHLMMSEChannelEstimator(grid, pc.dmrs.length,
        pc.dmrs.additional_position, pc.dmrs.num_cdm_groups_without_data,
        cov.time, cov.frequency, cov.space, order='s-f-t', device='cpu')
    y = torch.randn(1, 1, adapter.num_rx_antennas, grid.num_ofdm_symbols,
                    grid.fft_size, dtype=torch.complex64)
    expected = native(y, torch.tensor(0.1))
    actual = _WindowedLMMSE(adapter, cov, window_prbs)(y, torch.tensor(0.1))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


def test_covariance_sampling_is_finite_and_cropped():
    from neural_rx.classical import estimate_covariance

    adapter = SionnaPuschAdapter.from_config(
        Path(__file__).parents[1] / 'config/nrx_rt.cfg', device='cpu')
    cov = estimate_covariance(adapter, num_samples=1, frequency_size=12)
    assert cov.frequency.shape == (12, 12)
    for matrix in (cov.time, cov.frequency, cov.space):
        assert torch.isfinite(matrix).all()
        torch.testing.assert_close(matrix, matrix.mH)


def test_covariance_uses_umi_indoor_population_without_changing_evaluation():
    from neural_rx.classical import estimate_covariance
    from neural_rx.phy import _single_sector_topology
    from neural_rx.reproducibility import seed_everything

    seed_everything(9)
    adapter = SionnaPuschAdapter.from_config(
        Path(__file__).parents[1] / 'config/nrx_rt.cfg', device='cpu')
    estimate_covariance(adapter, num_samples=1, batch_size=16, frequency_size=12)
    # Covariance sampling leaves indoor_probability at UMi's 0.8 default;
    # evaluation explicitly sets it to zero.
    assert adapter.channel_model._scenario._in_state.any()
    evaluation_topology = _single_sector_topology(16, 1, 0., 1., adapter.device)
    assert not evaluation_topology[-1].any()
