"""Validate the real coded-signal boundary without requiring TensorRT."""

from pathlib import Path

import torch

from neural_rx import NeuralReceiver, SionnaPuschAdapter, load_experiment_config
from experiments.tensorrt_experiment import (
    AerialDataGenerator, AerialReceiverWithChannel, TrainingAlignedAerialReceiver,
)


def test_true_sparse_pilots_and_two_output_receiver():
    path = Path(__file__).parents[1] / "config/nrx_rt.cfg"
    adapter = SionnaPuschAdapter.from_config(path, n_size_bwp_override=2, device="cpu")
    generator = AerialDataGenerator(adapter)
    inputs, batch = generator(batch_size=1)
    assert inputs[2].shape == (1, 24, 2, 4)
    assert inputs[5].tolist() == [[2, 11], [2, 11]]
    assert inputs[6].shape == (2, 6)
    receiver = NeuralReceiver(load_experiment_config(path).receiver).eval()
    wrapper = AerialReceiverWithChannel(receiver, 2)
    with torch.no_grad():
        llrs, channel = wrapper(*inputs)
        received, pilots, initial = wrapper.prepare_inputs(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[5], inputs[6])
        result = receiver(received, pilots, inputs[4], initial)[-1]
    torch.testing.assert_close(llrs, -result.llrs.permute(0, 4, 1, 2, 3))
    torch.testing.assert_close(channel, result.channel)
    assert torch.isfinite(llrs).all()
    assert torch.isfinite(channel).all()
    assert batch.information_bits.shape[:2] == (1, 2)


def test_training_aligned_boundary_matches_native_preprocessing_and_logits():
    path = Path(__file__).parents[1] / "config/nrx_rt.cfg"
    adapter = SionnaPuschAdapter.from_config(path, n_size_bwp_override=3, device="cpu")
    inputs, batch = AerialDataGenerator(adapter)(batch_size=2)
    receiver = NeuralReceiver(load_experiment_config(path).receiver).eval()
    wrapper = TrainingAlignedAerialReceiver(receiver, adapter)
    received, features, initial = wrapper.prepare_inputs(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[5], inputs[6])
    torch.testing.assert_close(received, batch.received)
    torch.testing.assert_close(features, batch.pilot_features)
    torch.testing.assert_close(initial, batch.initial_channel, atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        llrs, channel = wrapper(*inputs)
        expected = receiver(batch.received, batch.pilot_features,
                            batch.active_users, batch.initial_channel)[-1]
    torch.testing.assert_close(llrs, -expected.llrs.permute(0, 4, 1, 2, 3),
                               atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(channel, expected.channel, atol=1e-6, rtol=1e-5)


def test_gather_index_reproduces_the_reference_rearrangement():
    path = Path(__file__).parents[1] / "config/nrx_rt.cfg"
    adapter = SionnaPuschAdapter.from_config(path, n_size_bwp_override=2, device="cpu")
    receiver = NeuralReceiver(load_experiment_config(path).receiver).eval()
    wrapper = TrainingAlignedAerialReceiver(receiver, adapter)
    inputs, _ = AerialDataGenerator(adapter)(batch_size=1)
    received, pilots, channel = wrapper.prepare_inputs(
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[5], inputs[6])
    expected, positions = receiver.initializer_features(received, pilots, channel)
    source = torch.cat((
        received.flatten(1),
        wrapper._average_focc(torch.cat((inputs[2], inputs[3]), dim=-1)).flatten(1),
        wrapper.pilot_source,
    ), dim=1)

    torch.testing.assert_close(
        source[:, wrapper.feature_index].reshape(expected.shape), expected)
    torch.testing.assert_close(wrapper.feature_pilots, positions)
