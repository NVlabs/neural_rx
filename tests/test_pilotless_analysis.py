"""Controlled retransmission checks for the original pilotless experiment."""

from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import pytest
import torch

from neural_rx.phy import SionnaPuschAdapter


ROOT = Path(__file__).parents[1]


def test_fixed_slot_exclusion_and_real_feature_mse(monkeypatch):
    import experiments.pilotless_analysis as analysis

    adapter = SionnaPuschAdapter.from_config(ROOT / "config/e2e_large.cfg")
    channel = adapter.channels[0]
    transmitted, targets, observations, noises = [], [], [], []

    def capture_channel(x):
        y, h = channel(x)
        transmitted.append(x.clone())
        targets.append(adapter._effective_channel(h, 0, adapter.pusch_configs[0]))
        return y, h

    awgn = adapter.awgn

    def capture_noise(y, no):
        noises.append(float(no))
        return awgn(y, no)

    class ZeroReceiver(torch.nn.Module):
        def forward(self, received, pilot_features, active_users, initial_channel=None,
                    *, mcs_mask=None):
            observations.append(received.clone())
            return (SimpleNamespace(channel=torch.zeros_like(received[:, None])),)

    monkeypatch.setattr(adapter, "channels", [capture_channel])
    monkeypatch.setattr(adapter, "awgn", capture_noise)
    result = analysis.channel_estimation_experiment(
        adapter, ZeroReceiver(), batch_size=3, num_trials=2, esno_db=6,
        excluded_symbols=[0], seed=17, microbatch_size=2,
    )
    assert [len(x) for x in transmitted] == [2, 1, 2, 1]
    for x in transmitted:
        torch.testing.assert_close(x, result["transmitted_grid"].expand_as(x))
    data = adapter.data_mask[0].T
    points = adapter.constellation_points.detach() / 2**0.5
    samples = result["transmitted_grid"][0, 0, 0][data]
    assert not ((samples - points[0]).abs() < 1e-6).any()
    assert ((samples[:, None] - points[None, 1:]).abs() < 1e-6).any(-1).all()
    expected = torch.cat(targets).square().mean(dim=(0, -1)).unsqueeze(0)
    torch.testing.assert_close(result["mse_grid"], expected)
    torch.testing.assert_close(result["mean_mse"], expected.mean())
    assert result["samples"] == 6
    assert noises == pytest.approx([10**(-0.6)] * 4)
    for y in observations:
        assert torch.count_nonzero(y[:, adapter.pilot_mask[0]]) == 0


def test_exclusion_preserves_other_data_and_pilots():
    from experiments.pilotless_analysis import channel_estimation_experiment
    from neural_rx.config import load_receiver_config
    from neural_rx.model import NeuralReceiver

    path = ROOT / "config/e2e_rt.cfg"
    adapter = SionnaPuschAdapter.from_config(path)
    model = NeuralReceiver(load_receiver_config(path)).eval()
    options = dict(batch_size=1, num_trials=1, esno_db=6, seed=12)
    original = channel_estimation_experiment(adapter, model, excluded_symbols=[], **options)
    changed = channel_estimation_experiment(adapter, model, excluded_symbols=[0], **options)
    x, z = original["transmitted_grid"], changed["transmitted_grid"]
    excluded = (x[0, 0, 0] - adapter.constellation_points[0] / 2**0.5).abs() < 1e-6
    assert excluded.any()
    torch.testing.assert_close(x[..., ~excluded], z[..., ~excluded])
    assert not torch.equal(x[..., excluded], z[..., excluded])


def test_rejects_empty_remaining_alphabet():
    from experiments.pilotless_analysis import channel_estimation_experiment

    adapter = SionnaPuschAdapter.from_config(ROOT / "config/e2e_rt.cfg")
    with pytest.raises(ValueError, match="remain"):
        channel_estimation_experiment(adapter, None, batch_size=1, num_trials=1,
                                      esno_db=6, excluded_symbols=list(range(16)))


def test_two_user_adapter_transmits_and_scores_only_user_zero():
    from experiments.pilotless_analysis import channel_estimation_experiment
    from neural_rx.config import load_receiver_config
    from neural_rx.model import NeuralReceiver

    path = ROOT / "config/nrx_rt.cfg"
    adapter = SionnaPuschAdapter.from_config(path)
    adapter.channel_estimator = None
    model = NeuralReceiver(replace(load_receiver_config(path),
                                   initial_channel_estimate=False)).eval()
    result = channel_estimation_experiment(
        adapter, model, batch_size=1, num_trials=1, esno_db=6,
        excluded_symbols=[0], seed=19,
    )
    assert result["transmitted_grid"].shape[1] == 2
    assert torch.count_nonzero(result["transmitted_grid"][:, 1]) == 0
    assert result["mse_grid"].shape == (1, 48, 14)
    assert torch.isfinite(result["mean_mse"])
