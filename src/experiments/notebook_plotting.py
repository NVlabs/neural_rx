"""Notebook figures, kept apart from the counted experiments so that
figure-only changes do not invalidate the measurement cache."""

import matplotlib.pyplot as plt
import torch

from experiments import notebook_experiments as experiments
from neural_rx.config import load_experiment_config
from neural_rx.evaluation import evaluation_ebno_points

LABELS = ['Neural Receiver', 'Baseline - LS/lin+LMMSE',
          'Baseline - LMMSE+K-Best', 'Baseline - Perf. CSI & K-Best']


def _plot(config_name, *, goodput=False, num_tx_eval=None, sim_idx=None, labels=None,
          fig=None, xlim=None, ylim=None, color_offset=0, mcs_arr_eval_idx=0,
          n_size_bwp=None, ebno_db=None):
    path = experiments._path(config_name)
    config = load_experiment_config(path)
    users = config.max_num_tx if num_tx_eval is None else num_tx_eval
    # Overlay calls run only the explicitly selected receiver curves.
    if sim_idx is None:
        sim_idx = (0,) if fig is not None else (0, 1, 2, 3)
    elif isinstance(sim_idx, int):
        sim_idx = (sim_idx,)
    if labels is None:
        labels = [LABELS[index] for index in sim_idx]
    if fig is None:
        fig, axis = plt.subplots(figsize=(12, 8))
    else:
        axis = fig.axes[0]
    for ordinal, index in enumerate(sim_idx):
        rows = experiments.run_bler(
            path, receiver=experiments.RECEIVERS[index], num_users=users,
            mcs_indices=[mcs_arr_eval_idx] * users, n_size_bwp=n_size_bwp,
            ebno_db=ebno_db)['results']
        colors = experiments.COLORMAP
        style = dict(label=labels[ordinal], linewidth=3.,
                     color=colors[(color_offset + (index if goodput else ordinal)) % len(colors)])
        if goodput:
            adapter = experiments.evaluation_adapter(path, users, n_size_bwp, experiments.DEVICE)
            bler = torch.tensor([row['bler'] for row in rows])
            piloted, pilotless = experiments.goodput(bler, adapter)
            y = pilotless if adapter.mask_pilots and index == 0 else piloted
            axis.plot([row['ebno_db'] for row in rows], y.numpy(), **style)
        else:
            experiments.plot_counted_bler(axis, rows, **style)
    mcs = list(config.receiver.mcs_indices)
    prbs = n_size_bwp or config.evaluation.n_size_bwp
    channel = config.evaluation.channel_type
    if users == 1 and channel.startswith('DoubleTDL'):
        channel = 'TDL-B100'
    axis.set_title(
        f"{'Goodput:' if goodput else '5G NR PUSCH'} "
        f'{users}x{config.receiver.num_rx_antennas} MU-MIMO, {channel}-Channel, '
        f'MCS={mcs if goodput else mcs[mcs_arr_eval_idx]}, PRBs={prbs}',
        fontsize=15,
    )
    axis.set_xlabel('SNR [dB]', fontsize=15)
    axis.set_ylabel('[info. bits / RE]' if goodput else 'TBLER', fontsize=15)
    axis.tick_params(axis='both', labelsize=15)
    if xlim is None:
        points = evaluation_ebno_points(path) if ebno_db is None else ebno_db
        if len(points) > 1:
            xlim = (min(points), max(points))
    if xlim is not None:
        axis.set_xlim(xlim)
    if ylim is not None:
        axis.set_ylim(ylim)
    axis.grid(True, which='both')
    axis.legend(loc='lower left', fontsize=15)
    return fig


def plot_results(config_name, **kwargs):
    """Plot the TBLER curves of ``config_name``, simulating missing results."""
    return _plot(config_name, **kwargs)


def plot_gp(config_name, **kwargs):
    """Plot the goodput of ``config_name``, simulating missing results."""
    return _plot(config_name, goodput=True, **kwargs)


def plot_finetuning_sweep(records):
    """Plot measured site-specific fine-tuning thresholds."""
    from matplotlib.lines import Line2D
    colors = {100: 'red', 200: 'blue', 1000: 'green', 10000: 'orange'}
    labels = {100: '100', 200: '200', 1000: '10^3', 10000: '10^4'}
    figure, axis = plt.subplots(figsize=(12, 6.15))
    with plt.rc_context({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'}):
        for size, color in colors.items():
            for channel, style in (('site', '-'), ('umi', ':')):
                rows = sorted((row for row in records if row['dataset_size'] == size
                               and row['channel'] == channel and row['steps']),
                              key=lambda row: row['steps'])
                axis.semilogx([row['steps'] for row in rows],
                              [row['snr_db'] if row['snr_db'] is not None else float('nan')
                               for row in rows], color=color, linestyle=style, linewidth=2.)
        for channel, style in (('site', '-'), ('umi', ':')):
            row = next((row for row in records if row['steps'] == 0
                        and row['channel'] == channel), None)
            if row is not None and row['snr_db'] is not None:
                axis.axhline(row['snr_db'], color='black', linestyle=style, linewidth=2.)
        omitted = sum(row['snr_db'] is None for row in records)
        if omitted:
            axis.text(.01, .01, f'{omitted} censored thresholds omitted; see threshold records',
                      transform=axis.transAxes, fontsize='small')
        handles = [Line2D([], [], color='black', linewidth=2., label='Generalized')]
        handles.extend(Line2D([], [], color=color, linewidth=2.,
                              label=rf'$N_{{\mathrm{{TD}}}}={labels[size]}$')
                       for size, color in colors.items())
        first = axis.legend(handles=handles, loc='upper right', fontsize=14)
        axis.add_artist(first)
        axis.legend(handles=[Line2D([], [], color='black', linestyle=style,
                                    linewidth=2., label=label)
                             for style, label in (('-', 'site-specific'), (':', '3GPP UMi'))],
                    loc='lower center', fontsize=14)
        axis.set_xlabel(r'Number of finetuning iterations $N_{\mathrm{FT}}$',
                        fontsize=16, fontfamily='serif', math_fontfamily='dejavuserif')
        axis.set_ylabel('SNR@TBLER=10% [dB]', fontsize=16, fontfamily='serif')
        axis.set_xlim(100, 100000)
        axis.tick_params(axis='both', labelsize=14)
        for label in axis.get_xticklabels() + axis.get_yticklabels():
            label.set_fontfamily('serif')
        for spine in axis.spines.values():
            spine.set_linewidth(1.5)
        axis.grid(True, which='both')
    return figure
