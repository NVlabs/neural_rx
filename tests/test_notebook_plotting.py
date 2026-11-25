import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest


@pytest.mark.parametrize('goodput', [False, True])
def test_figure_text_and_style_preserve_measurements(monkeypatch, goodput):
    from experiments import notebook_experiments as experiments
    from experiments import notebook_plotting as plotting
    monkeypatch.setattr(experiments.SionnaPuschAdapter, 'from_config', lambda *a, **k: object())
    monkeypatch.setattr(experiments, 'run_bler', lambda *a, **k: {
        'results': [dict(ebno_db=0., bler=.1, block_errors=10, blocks=100),
                    dict(ebno_db=1., bler=.01, block_errors=1, blocks=100)]})
    monkeypatch.setattr(experiments, 'goodput', lambda y, adapter: (1-y, 1-y))
    if goodput:
        from types import SimpleNamespace
        monkeypatch.setattr(experiments.SionnaPuschAdapter, 'from_config',
                            lambda *a, **k: SimpleNamespace(mask_pilots=False))
    function = plotting.plot_gp if goodput else plotting.plot_results
    fig = function('e2e_rt.cfg', sim_idx=(0,), num_tx_eval=1,
                   color_offset=5, labels=['Pilotless: NRX large'])
    ax = fig.axes[0]
    expected = ('Goodput: 1x4 MU-MIMO, TDL-B100-Channel, MCS=[14], PRBs=132'
                if goodput else '5G NR PUSCH 1x4 MU-MIMO, TDL-B100-Channel, MCS=14, PRBs=132')
    assert ax.get_title() == expected
    assert ax.get_xlabel() == 'SNR [dB]'
    assert ax.get_ylabel() == ('[info. bits / RE]' if goodput else 'TBLER')
    assert ax.lines[0].get_color() == '#e48aa7'
    assert ax.lines[0].get_linewidth() == 3
    assert ax.lines[0].get_marker() in ('None', '', None)
    assert ax.title.get_fontsize() == 15
    assert ax.legend_.texts[0].get_fontsize() == 15
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [.9, .99] if goodput else [.1, .01])
    plt.close(fig)


def test_goodput_subset_keeps_receiver_colors(monkeypatch):
    from types import SimpleNamespace
    from experiments import notebook_experiments as experiments
    from experiments.notebook_plotting import plot_gp
    monkeypatch.setattr(experiments.SionnaPuschAdapter, 'from_config',
                        lambda *a, **k: SimpleNamespace(mask_pilots=False))
    monkeypatch.setattr(experiments, 'run_bler', lambda *a, **k: {
        'results': [dict(ebno_db=0., bler=.1, block_errors=1, blocks=10)]})
    monkeypatch.setattr(experiments, 'goodput', lambda y, adapter: (1-y, 1-y))
    fig = plot_gp('nrx_rt.cfg', sim_idx=[1, 2, 3], n_size_bwp=132)
    assert [line.get_color() for line in fig.axes[0].lines] == ['#F3BD00', '#814B9D', '#5C5C5C']
    assert [line.get_label() for line in fig.axes[0].lines] == [
        'Baseline - LS/lin+LMMSE', 'Baseline - LMMSE+K-Best', 'Baseline - Perf. CSI & K-Best']
    plt.close(fig)


def test_site_sweep_uses_two_legends_without_changing_thresholds():
    from experiments.notebook_plotting import plot_finetuning_sweep as plot
    rows = [dict(dataset_size=size, channel=channel, steps=step,
                 snr_db=(None if step == 1000 else 13.))
            for size in (100, 200, 1000, 10000)
            for channel in ('site', 'umi') for step in (100, 1000)]
    rows += [dict(dataset_size=0, channel=channel, steps=0, snr_db=14.)
             for channel in ('site', 'umi')]
    fig = plot(rows)
    ax = fig.axes[0]
    assert ax.get_xlabel() == r'Number of finetuning iterations $N_{\mathrm{FT}}$'
    assert ax.get_ylabel() == 'SNR@TBLER=10% [dB]'
    assert [line.get_color() for line in ax.lines[:8:2]] == ['red', 'blue', 'green', 'orange']
    assert all(line.get_marker() in ('None', '', None) for line in ax.lines)
    assert ax.lines[0].get_ydata()[0] == 13.
    assert np.isnan(ax.lines[0].get_ydata()[1])
    assert ax.get_ylim()[1] > 14.  # Do not clip measured results.
    from matplotlib.legend import Legend
    legends = [artist for artist in ax.get_children() if isinstance(artist, Legend)]
    assert len(legends) == 2
    assert [text.get_text() for text in legends[-1].texts] == ['site-specific', '3GPP UMi']
    assert ax.get_xscale() == 'log'
    assert ax.get_xlim() == (100., 100000.)
    plt.close(fig)


def test_one_user_title_names_resolved_channel_and_snr_limits(monkeypatch):
    from experiments import notebook_experiments as experiments
    from experiments.notebook_plotting import plot_results
    monkeypatch.setattr(experiments.SionnaPuschAdapter, 'from_config', lambda *a, **k: object())
    monkeypatch.setattr(experiments, 'run_bler', lambda *a, **k: {
        'results': [dict(ebno_db=0., bler=.1, block_errors=1, blocks=10)]})
    fig = plot_results('nrx_large.cfg', sim_idx=0, num_tx_eval=1, ebno_db=[-2., 0., 7.])
    assert 'TDL-B100-Channel' in fig.axes[0].get_title()
    assert fig.axes[0].get_xlim() == (-2., 7.)
    plt.close(fig)


@pytest.mark.parametrize('site', [False, True])
def test_lazy_inline_backend_activation_is_not_rolled_back(monkeypatch, site):
    from experiments import notebook_experiments as experiments
    from experiments.notebook_plotting import plot_results, plot_finetuning_sweep
    monkeypatch.setattr(experiments.SionnaPuschAdapter, 'from_config', lambda *a, **k: object())
    monkeypatch.setattr(experiments, 'run_bler', lambda *a, **k: {
        'results': [dict(ebno_db=0., bler=.1, block_errors=1, blocks=10)]})
    subplots = plt.subplots
    def activate_inline(*args, **kwargs):
        # IPython's lazy backend integration activates interactivity on the
        # first figure. A style context must not roll that activation back.
        plt.ion()
        return subplots(*args, **kwargs)
    monkeypatch.setattr(plt, 'subplots', activate_inline)
    before = plt.isinteractive()
    plt.ioff()
    try:
        plot_finetuning_sweep([]) if site else plot_results('nrx_rt.cfg', sim_idx=0)
        assert plt.isinteractive()
    finally:
        plt.close('all')
        plt.interactive(before)
