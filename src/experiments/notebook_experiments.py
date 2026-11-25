"""Counted Monte Carlo experiments used by the notebooks.

The default budgets match the full evaluation settings. Results are stored in
``results/`` keyed by the experiment settings only; pass ``force=True`` to
re-simulate after changing code or checkpoints.
Each SNR point is persisted so interrupted long experiments can be resumed.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from neural_rx.classical import (
    ChannelCovariance, ClassicalReceiver, estimate_covariance, lmmse_window_prbs,
)
from neural_rx.config import load_experiment_config
from neural_rx.evaluation import evaluation_ebno_points
from neural_rx.model import NeuralReceiver
from neural_rx.phy import SionnaPuschAdapter
from neural_rx.reproducibility import seed_everything
from neural_rx.training import load_checkpoint

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / 'results'
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
RECEIVERS = ('neural', 'ls_linear_lmmse', 'lmmse_kbest', 'perfect_kbest')
COLORMAP = ('#76B900', '#F3BD00', '#814B9D', '#5C5C5C', '#214B9D',
            '#e48aa7', '#7c4848', '#78A2EB', '#ff1397', '#bee7dd')


def _path(config_name):
    path = Path(config_name)
    return path if path.is_file() else ROOT / 'config' / path


def _checkpoint(config_path, checkpoint_name=None):
    if checkpoint_name is not None:
        path = Path(checkpoint_name)
        if path.is_file():
            return path
        return ROOT / 'weights/pretrained' / (path.stem + '.safetensors')
    name = config_path.stem
    name = {'nrx_site_specific_baseline': 'nrx_rt',
            'nrx_site_specific_baseline_large': 'nrx_large'}.get(name, name)
    return ROOT / 'weights/pretrained' / f'{name}.safetensors'


def load_model(config_name, checkpoint_name=None, *, mode='evaluation',
               num_users=None, n_size_bwp=None, device=DEVICE):
    path = _path(config_name)
    experiment = load_experiment_config(path)
    adapter = SionnaPuschAdapter.from_config(
        path, mode=mode, device=str(device), num_users_override=num_users,
        n_size_bwp_override=n_size_bwp,
    )
    model = NeuralReceiver(experiment.receiver).to(device).eval()
    checkpoint = _checkpoint(path, checkpoint_name)
    load_checkpoint(model, checkpoint, device=device,
                    extra_tensors=adapter.checkpoint_tensors)
    return experiment, adapter, model


def goodput(
    bler: float | torch.Tensor,
    adapter: SionnaPuschAdapter,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return piloted and pilotless goodput for a PUSCH resource grid."""

    error_rate = torch.as_tensor(bler)
    successful_bits = (1.0 - error_rate) * adapter.pusch_configs[0].tb_size
    resource_grid = adapter.resource_grid
    resource_elements = math.prod(resource_grid.build_type_grid().shape[-2:])
    pilots = adapter.transmitter.pilot_pattern.pilots[0]
    nonempty_pilots = int((pilots.abs() > 1e-6).sum().item())
    empty_pilots = int((pilots.abs() < 1e-6).sum().item())
    piloted = successful_bits / (resource_elements - empty_pilots)
    pilotless = successful_bits / (
        resource_elements - nonempty_pilots - empty_pilots
    )
    return piloted, pilotless


def _name(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name


@functools.lru_cache(maxsize=None)
def evaluation_adapter(config_path, num_users=None, n_size_bwp=None, device=DEVICE):
    """Shared adapter for read-only metadata; simulations build their own."""
    return SionnaPuschAdapter.from_config(
        config_path, mode='evaluation', device=device,
        num_users_override=num_users, n_size_bwp_override=n_size_bwp,
    )


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def _covariance(config_path, adapter, *, samples, cache_dir, seed):
    # Covariance uses normalized UMi with the evaluation velocity range.
    experiment = load_experiment_config(config_path)
    covariance_adapter = SionnaPuschAdapter.from_config(
        config_path, mode='training', training_channel_type='UMi',
        n_size_bwp_override=adapter.resource_grid.fft_size // 12,
        num_users_override=1,
        device=str(adapter.device),
    )
    covariance_adapter.min_ut_velocity = experiment.evaluation.min_ut_velocity
    covariance_adapter.max_ut_velocity = experiment.evaluation.max_ut_velocity
    grid = adapter.resource_grid
    spec = dict(samples=samples, batch_size=30, seed=seed,
                fft_size=grid.fft_size, symbols=grid.num_ofdm_symbols,
                num_rx_antennas=adapter.num_rx_antennas,
                carrier_frequency=adapter.carrier_frequency,
                subcarrier_spacing=float(grid.subcarrier_spacing),
                num_tx_antennas=adapter.pusch_configs[0].num_antenna_ports,
                min_speed=experiment.evaluation.min_ut_velocity,
                max_speed=experiment.evaluation.max_ut_velocity)
    path = cache_dir / ('covariance-' + _digest(spec)[:24] + '.safetensors')
    if path.exists():
        values = load_file(str(path), device=str(adapter.device))
        return ChannelCovariance(values['time'], values['frequency'], values['space'])
    seed_everything(seed)
    covariance = estimate_covariance(covariance_adapter, samples, batch_size=30,
        frequency_size=12 * lmmse_window_prbs(grid.fft_size // 12))
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: getattr(covariance, name).contiguous()
               for name in ('time', 'frequency', 'space')}, str(path),
              metadata={'provenance': json.dumps(spec, sort_keys=True)})
    return covariance


@torch.inference_mode()
def run_bler(config_name, *, receiver='neural', checkpoint_name=None,
             num_users=None, mcs_indices=None, report_user=None,
             ebno_db=None, batch_size=None, max_batches=500,
             target_block_errors=500, target_bler=0.001,
             num_iterations=None, covariance_samples=100_000,
             seed=42, device=DEVICE, n_size_bwp=None, cache_dir=None,
             force=False):
    """Measure BER and BLER sweeps without synthesizing or interpolating curves."""
    if receiver not in RECEIVERS:
        raise ValueError(f'Unknown receiver: {receiver}')
    if max_batches < 1:
        raise ValueError('max_batches must be positive')
    config_path = _path(config_name)
    experiment = load_experiment_config(config_path)
    adapter = evaluation_adapter(config_path, num_users, n_size_bwp, str(device))
    indices = list(mcs_indices) if mcs_indices is not None else [0] * adapter.num_users
    if len(indices) != adapter.num_users or any(
        not 0 <= index < len(adapter.num_bits_by_mcs) for index in indices
    ):
        raise ValueError('Provide one valid MCS-array index per user')
    if report_user is not None and not 0 <= report_user < adapter.num_users:
        raise ValueError('report_user is outside the configured users')
    snr_mcs_index = indices[0 if report_user is None else report_user]
    points = evaluation_ebno_points(config_path) if ebno_db is None else list(map(float, ebno_db))
    batch_size = (3 if 'kbest' in receiver else experiment.evaluation.batch_size) if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    checkpoint = _checkpoint(config_path, checkpoint_name) if receiver == 'neural' else None
    spec = dict(config=config_path.name, receiver=receiver,
                checkpoint=_name(checkpoint) if checkpoint else None,
                num_users=adapter.num_users, n_size_bwp=adapter.resource_grid.fft_size // 12,
                mcs_indices=indices, snr_mcs_index=snr_mcs_index, report_user=report_user,
                ebno_db=points, batch_size=batch_size, max_batches=max_batches,
                target_block_errors=target_block_errors, target_bler=target_bler,
                num_iterations=num_iterations, covariance_samples=covariance_samples,
                seed=seed)
    cache_dir = RESULTS if cache_dir is None else Path(cache_dir)
    path = cache_dir / (config_path.stem + '-' + receiver + '-' + _digest(spec)[:12] + '.json')
    payload = dict(spec=spec, results=[], complete=False)
    if path.exists() and not force:
        payload = json.loads(path.read_text())
        if payload.get('complete'):
            return payload
        if (payload['results'] and target_bler is not None
                and payload['results'][-1]['bler'] <= target_bler):
            payload['complete'] = True
            _write(path, payload)
            return payload
    model = classical = None
    if receiver == 'neural':
        _, adapter, model = load_model(config_path, checkpoint, num_users=adapter.num_users,
                                       n_size_bwp=n_size_bwp, device=device)
        if num_iterations is not None:
            if not 1 <= num_iterations <= len(model.iterations):
                raise ValueError('Iteration count must select a trained prefix')
            model.iterations = model.iterations[:num_iterations]
    else:
        adapter = SionnaPuschAdapter.from_config(
            config_path, mode='evaluation', device=str(device),
            num_users_override=num_users, n_size_bwp_override=n_size_bwp,
        )
        covariance = (_covariance(config_path, adapter, samples=covariance_samples,
                                  cache_dir=cache_dir, seed=seed + 1_000_000)
                      if receiver == 'lmmse_kbest' else None)
        if len(set(indices)) > 1 and receiver != 'ls_linear_lmmse':
            raise ValueError('Mixed constellations require the linear baseline, not homogeneous K-best')
        classical = {index: ClassicalReceiver(adapter, receiver, covariance, mcs_index=index)
                     for index in set(indices)}
    adapter.compile_decoders()
    selected = torch.tensor(indices, device=device)
    mask = torch.nn.functional.one_hot(selected, len(adapter.num_bits_by_mcs)).float()
    for point_index in range(len(payload['results']), len(points)):
        point = points[point_index]
        seed_everything(seed + point_index)
        started = time.monotonic()
        counts = [dict(user=user, mcs_array_index=index, bits=0, blocks=0,
                       bit_errors=0, block_errors=0, crc_successes=0)
                  for user, index in enumerate(indices)]
        for _ in range(max_batches):
            batch = adapter.sample(batch_size, point,
                mcs_mask=mask[None].expand(batch_size, -1, -1),
                snr_mcs_index=snr_mcs_index)
            information = ((batch.information_bits,) if isinstance(batch.information_bits, torch.Tensor)
                           else batch.information_bits)
            if model is not None:
                llrs = model.forward_llrs(batch.received, batch.pilot_features,
                    batch.active_users, batch.initial_channel, mcs_mask=batch.mcs_mask)
                decoded = {index: adapter.decode(llrs, index) for index in set(indices)}
            else:
                decoded = {index: detector(batch, adapter.noise_variance(point, mcs_index=snr_mcs_index))
                           for index, detector in classical.items()}
            for user, index in enumerate(indices):
                bits, crc = decoded[index]
                difference = bits[:, user] != information[index][:, user]
                count = counts[user]
                count['bits'] += difference.numel()
                count['blocks'] += difference.shape[0]
                count['bit_errors'] += int(difference.sum())
                count['block_errors'] += int(difference.any(dim=-1).sum())
                count['crc_successes'] += int(crc[:, user].sum())
            reported = counts if report_user is None else [counts[report_user]]
            if target_block_errors is not None and sum(c['block_errors'] for c in reported) >= target_block_errors:
                break
        for count in counts:
            count['ber'] = count['bit_errors'] / count['bits']
            count['bler'] = count['block_errors'] / count['blocks']
        row = {key: sum(count[key] for count in reported) for key in (
            'bits', 'blocks', 'bit_errors', 'block_errors', 'crc_successes')}
        row.update(ebno_db=point, ber=row['bit_errors'] / row['bits'],
                   bler=row['block_errors'] / row['blocks'], per_user=counts,
                   elapsed_seconds=time.monotonic() - started)
        payload['results'].append(row)
        _write(path, payload)
        print(f"{config_path.stem} {receiver} {point:g} dB: "
              f"{row['block_errors']}/{row['blocks']} blocks, BLER={row['bler']:.6g}", flush=True)
        if target_bler is not None and row['bler'] <= target_bler:
            break
    payload['complete'] = True
    _write(path, payload)
    return payload


def plot_counted_bler(axis, rows, *, x=None, **kwargs):
    """Plot measured block-error rates."""
    x = [row['ebno_db'] for row in rows] if x is None else list(x)
    line, = axis.semilogy(x, [row['bler'] for row in rows], **kwargs)
    return line
