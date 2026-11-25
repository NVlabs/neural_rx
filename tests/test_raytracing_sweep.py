from pathlib import Path

import pytest


def test_site_sweep_config_preserves_two_user_umi_control(tmp_path):
    from neural_rx.config import load_experiment_config
    from raytracing.sweep import make_sweep_config
    path = make_sweep_config(tmp_path, Path('/tmp/train.safetensors'),
                             Path('/tmp/eval.safetensors'), 200, 'umi')
    config = load_experiment_config(path)
    assert config.training.max_dataset_samples == 200
    assert config.training.phases[0].steps == 100_000
    assert config.evaluation.channel_type == 'UMi'
    assert not config.evaluation.normalize_channel
    assert config.evaluation.min_ut_velocity == 50 / 3.6
    assert config.evaluation.max_ut_velocity == 50 / 3.6


def test_threshold_interpolates_first_integer_bracket_in_log_bler():
    from raytracing.sweep import measured_threshold
    sampled = []
    def evaluate(snr):
        sampled.append(snr)
        return {'ebno_db': snr, 'bler': .2 if snr < 2.7 else .05}
    result = measured_threshold(evaluate, lower=0., upper=6.)
    assert result['snr_db'] == pytest.approx(2.5)
    assert sampled == [0., 1., 2., 3.]


@pytest.mark.parametrize('bler,censored', [
    (lambda snr: .2 if snr == 6 else 0., 'zero_endpoint'),
    (lambda snr: .2, 'above_range'),
    (lambda snr: .05, 'below_range'),
])
def test_unbracketed_thresholds_are_censored(bler, censored):
    from raytracing.sweep import measured_threshold
    result = measured_threshold(lambda snr: {'ebno_db': snr, 'bler': bler(snr)})
    assert result['snr_db'] is None
    assert result['censored'] == censored


def test_sweep_grid_has_26_checkpoints():
    from raytracing.sweep import SWEEP_GRID
    assert sum(len(set(channels['site']) | set(channels['umi']))
               for channels in SWEEP_GRID.values()) == 26


def test_sweep_measures_every_configured_endpoint(tmp_path, monkeypatch):
    import raytracing.sweep as sweep
    trained = []
    def train(config, initialization, output, steps, **kwargs):
        trained.append(tuple(steps))
        return {step: output.with_name(f'{output.stem}_steps{step}.safetensors') for step in steps}
    monkeypatch.setattr(sweep, '_train', train)
    monkeypatch.setattr(sweep, 'measured_threshold', lambda evaluate: {'snr_db': 9.})
    records = sweep.run_finetuning_sweep('train', 'eval', output_dir=tmp_path, device='cpu')
    measured = {(r['dataset_size'], r['channel'], r['steps']) for r in records if r['steps']}
    expected = {(size, channel, step) for size, channels in sweep.SWEEP_GRID.items()
                for channel, steps in channels.items() for step in steps}
    assert measured == expected
    assert sum(map(len, trained)) == 26


def test_saved_steps_match_independent_training_prefixes(tmp_path, monkeypatch):
    import ast
    import configparser
    import subprocess
    import sys
    import torch
    from safetensors.torch import load_file
    import raytracing.sweep as sweep

    monkeypatch.setenv('OMP_NUM_THREADS', '1')
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(sweep.ROOT / 'config/nrx_rt.cfg')
    schedule = ast.literal_eval(parser['training']['training_schedule'])
    schedule['batch_size'] = [1] * len(schedule['batch_size'])
    parser['training']['training_schedule'] = repr(schedule)
    config = tmp_path / 'tiny.cfg'
    with config.open('w') as stream:
        parser.write(stream)
    initialization = sweep.ROOT / 'weights/pretrained/nrx_rt.safetensors'
    saved = sweep._train(config, initialization, tmp_path / 'sweep.safetensors', (1, 2),
                         device='cpu', seed=42)
    for step, path in saved.items():
        independent = tmp_path / f'independent{step}.safetensors'
        subprocess.run([sys.executable, '-m', 'neural_rx.cli', 'train', '--config', str(config),
                        '--init-checkpoint', str(initialization), '--max-steps', str(step),
                        '--seed', '42', '--device', 'cpu', '--output', str(independent)],
                       check=True, cwd=sweep.ROOT, capture_output=True)
        actual, expected = load_file(path), load_file(independent)
        assert actual.keys() == expected.keys()
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
