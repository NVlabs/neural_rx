from pathlib import Path


def test_zero_errors_are_plotted_without_extra_annotations():
    import matplotlib.pyplot as plt
    from experiments.notebook_experiments import plot_counted_bler
    fig, axis = plt.subplots()
    rows = [dict(ebno_db=1., bler=.1, block_errors=10, blocks=100),
            dict(ebno_db=2., bler=0., block_errors=0, blocks=100)]
    plot_counted_bler(axis, rows, label='measured')
    assert list(axis.lines[0].get_ydata()) == [.1, 0.]
    assert not axis.containers
    plt.close(fig)


def test_plot_forwards_configured_bandwidth_and_snr_grid(monkeypatch):
    import matplotlib.pyplot as plt
    import experiments.notebook_experiments as experiments
    calls = []

    def measured(*args, **kwargs):
        calls.append(kwargs)
        return {'results': [{'ebno_db': 3., 'bler': .1, 'block_errors': 10, 'blocks': 100}]}

    monkeypatch.setattr(experiments, 'run_bler', measured)
    from experiments.notebook_plotting import plot_results
    figure = plot_results('nrx_rt.cfg', sim_idx=[0],
                                      n_size_bwp=132, ebno_db=[3.])
    assert calls[0]['n_size_bwp'] == 132
    assert calls[0]['ebno_db'] == [3.]
    plt.close(figure)


def test_counted_neural_sweep_and_exact_cache(tmp_path):
    from experiments.notebook_experiments import run_bler
    options = dict(ebno_db=[0., 1.], batch_size=1, max_batches=2,
                   target_block_errors=None, target_bler=None,
                   device='cpu', n_size_bwp=4, cache_dir=tmp_path)
    first = run_bler('nrx_rt.cfg', **options)
    assert first['complete'] is True
    assert len(first['results']) == 2
    for row in first['results']:
        assert row['blocks'] == 4
        assert row['bler'] == row['block_errors'] / row['blocks']
        assert sum(user['blocks'] for user in row['per_user']) == row['blocks']
    assert run_bler('nrx_rt.cfg', **options) == first
    assert len(list(tmp_path.glob('*.json'))) == 1


def test_mixed_sweep_reports_selected_user_with_correct_snr(tmp_path):
    from experiments.notebook_experiments import run_bler
    result = run_bler('nrx_rt_var_mcs.cfg', ebno_db=[3.],
                      mcs_indices=[1, 0], report_user=0, batch_size=1,
                      max_batches=1, device='cpu', n_size_bwp=4, cache_dir=tmp_path)
    assert result['spec']['snr_mcs_index'] == 1
    row = result['results'][0]
    assert row['blocks'] == 1
    assert row['bits'] == row['per_user'][0]['bits']


def test_perfect_csi_sweep_uses_real_decoder(tmp_path):
    from experiments.notebook_experiments import run_bler
    result = run_bler('nrx_rt.cfg', receiver='perfect_kbest', ebno_db=[30.],
                      batch_size=1, max_batches=1, device='cpu', n_size_bwp=4,
                      cache_dir=tmp_path)
    assert result['results'][0]['blocks'] == 2
    assert result['results'][0]['block_errors'] == 0


def test_mixed_linear_baseline_decodes_both_constellations(tmp_path):
    from experiments.notebook_experiments import run_bler
    result = run_bler('nrx_rt_var_mcs.cfg', receiver='ls_linear_lmmse',
                      ebno_db=[30.], mcs_indices=[1, 0], report_user=0,
                      batch_size=1, max_batches=1, device='cpu',
                      n_size_bwp=4, cache_dir=tmp_path)
    assert result['results'][0]['block_errors'] == 0
    assert len(result['results'][0]['per_user']) == 2


def test_resume_after_saved_early_stop_does_not_add_points(tmp_path, monkeypatch):
    import json
    import experiments.notebook_experiments as experiments
    options = dict(ebno_db=[0., 1.], batch_size=1, max_batches=1,
                   target_bler=1., device='cpu', n_size_bwp=4, cache_dir=tmp_path)
    result = experiments.run_bler('nrx_rt.cfg', **options)
    assert len(result['results']) == 1
    path = next(tmp_path.glob('*.json'))
    result['complete'] = False
    path.write_text(json.dumps(result))
    monkeypatch.setattr(experiments, 'load_model', lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError('A saved stopping condition must not restart simulation')))
    resumed = experiments.run_bler('nrx_rt.cfg', **options)
    assert resumed['complete']
    assert len(resumed['results']) == 1


def test_goodput_accounts_for_empty_pilot_resource_elements() -> None:
    import torch
    from experiments.notebook_experiments import goodput
    from neural_rx.phy import SionnaPuschAdapter
    bler = torch.tensor([0.0, 0.25, 1.0])
    adapter = SionnaPuschAdapter.from_config(
        Path(__file__).parents[1] / "config" / "nrx_rt.cfg",
        device="cpu",
    )

    piloted, pilotless = goodput(bler, adapter)

    pilots = adapter.transmitter.pilot_pattern.pilots[0]
    assert int((pilots.abs() < 1e-6).sum()) == 48
    torch.testing.assert_close(
        piloted,
        torch.tensor([1256 / 624, 0.75 * 1256 / 624, 0.0]),
    )
    torch.testing.assert_close(
        pilotless,
        torch.tensor([1256 / 576, 0.75 * 1256 / 576, 0.0]),
    )
