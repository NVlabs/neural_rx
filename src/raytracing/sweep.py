"""Site-specific fine-tuning sweep over dataset sizes and training steps."""

from __future__ import annotations

import ast
import configparser
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from experiments.notebook_experiments import ROOT, DEVICE, RESULTS, run_bler

STEPS = (100, 300, 1000, 3100, 10000)
SWEEP_GRID = {
    100: dict(site=STEPS, umi=STEPS[:4]),
    200: dict(site=(*STEPS, 31600, 100000), umi=STEPS),
    1000: dict(site=(*STEPS, 31600, 100000), umi=(*STEPS, 31600)),
    10000: dict(site=(*STEPS, 31600, 100000), umi=(*STEPS, 31600)),
}


def make_sweep_config(directory, training_dataset, evaluation_dataset, size, channel):
    if channel not in ('site', 'umi') or size < 1:
        raise ValueError('Expected positive dataset size and site/umi channel')
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(ROOT / 'config/nrx_site_specific.cfg')
    parser['global']['label'] = repr(f'notebook_site_n{size}_{channel}')
    schedule = ast.literal_eval(parser['training']['training_schedule'])
    schedule['num_iter'] = [100_000]
    parser['training']['training_schedule'] = repr(schedule)
    directory = Path(directory).resolve()
    parser['training']['dataset_path'] = repr(os.path.relpath(Path(training_dataset).resolve(), directory))
    parser['training']['max_dataset_samples'] = str(size)
    parser['evaluation']['dataset_path_eval'] = repr(os.path.relpath(Path(evaluation_dataset).resolve(), directory))
    if channel == 'umi':
        parser['evaluation']['channel_type_eval'] = repr('UMi')
        parser['evaluation']['channel_norm_eval'] = 'False'
        parser['evaluation']['min_ut_velocity_eval'] = repr(50 / 3.6)
        parser['evaluation']['max_ut_velocity_eval'] = repr(50 / 3.6)
        del parser['evaluation']['dataset_path_eval']
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'notebook_site_n{size}_{channel}.cfg'
    with path.open('w', encoding='utf-8') as stream:
        parser.write(stream)
    return path


def measured_threshold(evaluate, *, lower=6., upper=20., target=.1):
    """Interpolate the first 1-dB-grid crossing of ``target`` linearly in log10(BLER).

    Returns ``snr_db=None`` with a ``censored`` reason when the crossing is
    below the range, above it, or ends at zero observed errors.
    """
    rows = []
    previous = None
    for tenth in range(round(lower * 10), round(upper * 10) + 1, 10):
        row = evaluate(tenth / 10)
        rows.append(row)
        if row['bler'] <= target:
            if previous is None:
                return dict(snr_db=None, censored='below_range', measurements=rows)
            if row['bler'] == 0:
                return dict(snr_db=None, censored='zero_endpoint', measurements=rows)
            fraction = ((math.log10(target) - math.log10(previous['bler']))
                        / (math.log10(row['bler']) - math.log10(previous['bler'])))
            snr = previous['ebno_db'] + fraction * (row['ebno_db'] - previous['ebno_db'])
            return dict(snr_db=snr, censored=None, measurements=rows)
        previous = row
    return dict(snr_db=None, censored='above_range', measurements=rows)


def _train(config, initialization, output, steps, *, device, seed):
    """Fine-tune once and save a checkpoint at every requested step."""
    paths = {step: output.with_name(f'{output.stem}_steps{step}{output.suffix}') for step in steps}
    if all(path.exists() for path in paths.values()):
        return paths
    command = [sys.executable, '-m', 'neural_rx.cli', 'train', '--config', str(config),
               '--output', str(output), '--init-checkpoint', str(initialization),
               '--device', str(device), '--max-steps', str(max(steps)), '--seed', str(seed),
               '--save-steps', *map(str, steps)]
    if str(device).startswith('cuda'):
        command.append('--compile-model')
    subprocess.run(command, cwd=ROOT, check=True)
    return paths


def run_finetuning_sweep(training_dataset, evaluation_dataset, *,
                         device=DEVICE, seed=42, training_seeds=None, output_dir=None,
                         force=False):
    """Fine-tune the generalized receiver and measure site and UMi thresholds.

    ``training_seeds`` optionally maps dataset sizes to fine-tuning seeds;
    evaluation always uses ``seed``. The thresholds are stored in
    ``results/site_sweep.json`` (or ``output_dir``) and reused unless ``force`` is set.
    """
    results_path = Path(output_dir or RESULTS) / 'site_sweep.json'
    if results_path.exists() and not force:
        return json.loads(results_path.read_text())
    training_seeds = training_seeds or {}
    initialization = ROOT / 'weights/pretrained/nrx_rt.safetensors'
    directory = Path(output_dir or ROOT / 'weights/runs/site_sweep')
    records = []
    baseline = {}
    for size, grid in SWEEP_GRID.items():
        configs = {channel: make_sweep_config(directory, training_dataset, evaluation_dataset, size, channel)
                   for channel in grid}
        steps = sorted(set(grid['site']) | set(grid['umi']))
        training_seed = training_seeds.get(size, seed)
        name = f'n{size}' if training_seed == seed else f'n{size}_seed{training_seed}'
        checkpoints = _train(configs['site'], initialization, directory / f'{name}.safetensors',
                             steps, device=device, seed=training_seed)
        for step in (0, *steps):
            checkpoint = checkpoints.get(step, initialization)
            for channel, config in configs.items():
                if step and step not in grid[channel]:
                    continue
                if step == 0 and channel in baseline:
                    threshold = baseline[channel]
                else:
                    threshold = measured_threshold(lambda snr: run_bler(
                        config, checkpoint_name=checkpoint, num_users=2, ebno_db=[snr],
                        target_bler=None, seed=seed, device=device,
                        cache_dir=directory / 'evaluations')['results'][0])
                    if step == 0:
                        baseline[channel] = threshold
                records.append(dict(dataset_size=size, steps=step, channel=channel, **threshold))
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    return records
