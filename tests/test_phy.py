# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from pathlib import Path

import pytest
import torch

from neural_rx.config import load_receiver_config
from neural_rx.model import NeuralReceiver
from neural_rx.phy import SionnaPuschAdapter, sample_num_active_users
from neural_rx.cir_dataset import CIRDataset, save_cir_dataset
from neural_rx.training import receiver_loss


REPOSITORY_ROOT = Path(__file__).parents[1]


def test_sionna_pusch_batch_trains_receiver() -> None:
    config_path = REPOSITORY_ROOT / "config" / "nrx_rt.cfg"
    config = load_receiver_config(config_path)
    adapter = SionnaPuschAdapter.from_config(config_path, device="cpu")

    assert adapter.randomize_pilots is True
    assert adapter.pilot_set.shape[0] == 20
    assert not torch.equal(adapter.pilot_set[0], adapter.pilot_set[1])
    batch = adapter.sample(batch_size=1, ebno_db=8.0, num_active_users=1)

    assert adapter.channel_type == "UMi"
    assert batch.received.shape == (1, 48, 14, 8)
    assert batch.pilot_features.shape == (2, 48, 14, 2)
    assert batch.initial_channel.shape == (1, 2, 48, 14, 8)
    assert not torch.allclose(batch.initial_channel, batch.channel_target)
    assert batch.bit_targets.shape == (1, 2, 48, 14, 4)
    assert batch.data_mask.shape == (2, 48, 14)
    assert batch.data_mask.sum(dim=(1, 2)).tolist() == [576, 576]
    assert batch.active_users.tolist() == [[1.0, 0.0]]
    assert torch.isfinite(batch.received).all()


    ideal_llrs = (2 * batch.bit_targets - 1) * 20
    decoded, crc_status = adapter.decode(ideal_llrs)
    torch.testing.assert_close(decoded, batch.information_bits)
    assert crc_status.all()

    model = NeuralReceiver(config)
    outputs = model(
        batch.received,
        batch.pilot_features,
        batch.active_users,
        batch.initial_channel,
    )
    losses = receiver_loss(
        outputs,
        batch.bit_targets,
        batch.active_users,
        data_mask=batch.data_mask,
    )
    losses.total.backward()

    assert torch.isfinite(losses.total)
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_dataset_override_does_not_require_changing_config(tmp_path):
    dataset_path = tmp_path / 'tutorial.safetensors'
    save_cir_dataset(dataset_path, CIRDataset(
        torch.ones((4, 1, 4, 1, 2, 1, 14), dtype=torch.complex64),
        torch.zeros((4, 1, 1, 1), dtype=torch.float32),
    ))
    adapter = SionnaPuschAdapter.from_config(
        REPOSITORY_ROOT / 'config/nrx_site_specific.cfg',
        dataset_path_override=dataset_path, device='cpu',
    )
    assert torch.isfinite(adapter.sample(1, 8.0).received).all()


def test_single_user_evaluation_uses_tdl_b_fallback():
    adapter = SionnaPuschAdapter.from_config(
        REPOSITORY_ROOT / 'config/nrx_rt.cfg', mode='evaluation', device='cpu',
        num_users_override=1, n_size_bwp_override=4,
    )
    assert adapter.num_users == 1
    assert adapter.channel_type == 'TDL-B100'
    batch = adapter.sample(1, 8.0)
    assert batch.received.shape == (1, 48, 14, 8)
    assert batch.active_users.shape == (1, 1)


def test_bandwidth_override_preserves_two_user_channel():
    adapter = SionnaPuschAdapter.from_config(
        REPOSITORY_ROOT / 'config/nrx_rt.cfg', mode='evaluation', device='cpu',
        n_size_bwp_override=4,
    )
    assert adapter.num_users == 2
    assert adapter.channel_type == 'DoubleTDLlow'
    assert adapter.resource_grid.fft_size == 48


def test_noise_variance_uses_evaluated_mcs():
    from sionna.phy.utils import ebnodb2no
    adapter = SionnaPuschAdapter.from_config(
        REPOSITORY_ROOT / 'config/nrx_rt_var_mcs.cfg', device='cpu',
    )
    for index in (0, 1):
        expected = ebnodb2no(3., adapter.num_bits_by_mcs[index],
                            adapter.coderates[index], adapter.resource_grid, device='cpu')
        torch.testing.assert_close(adapter.noise_variance(3., mcs_index=index), expected)
    assert adapter.noise_variance(3., mcs_index=0) != adapter.noise_variance(3., mcs_index=1)


def test_evaluation_adapter_uses_configured_double_tdl_bandwidth() -> None:
    config_path = REPOSITORY_ROOT / "config" / "nrx_rt.cfg"

    adapter = SionnaPuschAdapter.from_config(config_path, mode="evaluation", device="cpu")

    assert adapter.channel_type == "DoubleTDLlow"
    assert adapter.resource_grid.fft_size == 273 * 12
    assert adapter.num_users == 2
    assert adapter.decoder_iterations == 20


def test_adapter_samples_configured_cir_dataset(tmp_path: Path) -> None:
    dataset_path = tmp_path / "training.safetensors"
    save_cir_dataset(
        dataset_path,
        CIRDataset(
            torch.ones((4, 1, 4, 1, 2, 1, 14), dtype=torch.complex64),
            torch.zeros((4, 1, 1, 1), dtype=torch.float32),
        ),
    )
    source = REPOSITORY_ROOT / "config" / "nrx_site_specific.cfg"
    config_path = tmp_path / source.name
    config_path.write_text(
        source.read_text(encoding="utf-8").replace(
                "../data/nrx_site_specific_train.safetensors", str(dataset_path)
        ),
        encoding="utf-8",
    )

    adapter = SionnaPuschAdapter.from_config(config_path, device="cpu")
    batch = adapter.sample(1, 8.0)

    assert adapter.channel_type == "Dataset"
    assert batch.received.shape == (1, 48, 14, 8)
    assert torch.isfinite(batch.received).all()


def test_e2e_adapter_trains_normalized_constellation_and_masks_pilots() -> None:
    config_path = REPOSITORY_ROOT / "config" / "e2e_rt.cfg"
    adapter = SionnaPuschAdapter.from_config(config_path, device="cpu")

    adapter.set_transmitter_trainable(True)
    points = adapter.constellation_points
    batch = adapter.sample(batch_size=1, ebno_db=3.5)

    assert adapter.channel_type == "TDL-C300"
    assert adapter.transmitter_parameters[0].requires_grad
    torch.testing.assert_close(points.mean(), torch.zeros_like(points.mean()), atol=1e-6, rtol=0)
    torch.testing.assert_close(points.abs().square().mean(), torch.tensor(1.0), atol=1e-6, rtol=0)
    assert batch.initial_channel is None
    assert torch.count_nonzero(batch.received[:, adapter.pilot_mask[0]]) == 0

    adapter.set_transmitter_trainable(False)
    assert not adapter.transmitter_parameters[0].requires_grad


def test_e2e_training_adapter_accepts_phase_channel_override() -> None:
    config_path = REPOSITORY_ROOT / "config" / "e2e_rt.cfg"

    adapter = SionnaPuschAdapter.from_config(
        config_path,
        training_channel_type="UMi",
        device="cpu",
    )

    assert adapter.channel_type == "UMi"
    assert len(adapter.transmitter_parameters) == 1


def test_active_user_sampler_matches_triangular_support() -> None:
    torch.manual_seed(3)
    samples = torch.stack([sample_num_active_users(1, 2) for _ in range(2000)])

    assert set(samples.tolist()) == {1, 2}
    assert (samples == 2).float().mean() > 0.7


def test_snr_mode_uses_direct_noise_variance(tmp_path: Path) -> None:
    source = REPOSITORY_ROOT / "config" / "nrx_rt.cfg"
    config_path = tmp_path / source.name
    config_path.write_text(
        source.read_text(encoding="utf-8").replace("ebno = True", "ebno = False"),
        encoding="utf-8",
    )
    adapter = SionnaPuschAdapter.from_config(config_path, device="cpu")

    class NoiseCapture:
        noise_variance: torch.Tensor | None = None

        def __call__(self, received: torch.Tensor, noise_variance: torch.Tensor) -> torch.Tensor:
            self.noise_variance = noise_variance
            return received

    capture = NoiseCapture()
    adapter.awgn = capture
    adapter.sample(batch_size=1, ebno_db=10.0, num_active_users=1)

    torch.testing.assert_close(capture.noise_variance, torch.tensor(0.1))


def test_mixed_mcs_batch_selects_per_user_transmitters_and_targets() -> None:
    config_path = REPOSITORY_ROOT / "config" / "nrx_rt_var_mcs.cfg"
    adapter = SionnaPuschAdapter.from_config(config_path, device="cpu")
    mcs_mask = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    batch = adapter.sample(1, 8.0, mcs_mask=mcs_mask)

    assert len(adapter.transmitters) == 2
    assert adapter.num_bits_by_mcs == (2, 4)
    assert isinstance(batch.information_bits, tuple)
    torch.testing.assert_close(batch.mcs_mask, mcs_mask)
    assert batch.bit_mask.tolist() == [[[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]]
    assert torch.count_nonzero(batch.bit_targets[:, 0, ..., 2:]) == 0

    ideal_llrs = (2 * batch.bit_targets - 1) * 20
    decoded_qpsk, crc_qpsk = adapter.decode(ideal_llrs, 0)
    decoded_qam, crc_qam = adapter.decode(ideal_llrs, 1)
    torch.testing.assert_close(decoded_qpsk[:, 0], batch.information_bits[0][:, 0])
    torch.testing.assert_close(decoded_qam[:, 1], batch.information_bits[1][:, 1])
    assert crc_qpsk[:, 0].all()
    assert crc_qam[:, 1].all()

    output = NeuralReceiver(load_receiver_config(config_path))(
        batch.received,
        batch.pilot_features,
        batch.active_users,
        batch.initial_channel,
        mcs_mask=batch.mcs_mask,
    )
    loss = receiver_loss(
        output,
        batch.bit_targets,
        batch.active_users,
        data_mask=batch.data_mask,
        bit_mask=batch.bit_mask,
    )
    loss.total.backward()
    assert torch.isfinite(loss.total)


def test_training_reuses_one_umi_draw_for_several_batches(monkeypatch) -> None:
    config_path = REPOSITORY_ROOT / "config" / "nrx_rt.cfg"
    adapter = SionnaPuschAdapter.from_config(config_path, device="cpu")
    evaluation = SionnaPuschAdapter.from_config(
        config_path, mode="evaluation", device="cpu", num_users_override=1
    )
    assert (adapter.channel_chunk, evaluation.channel_chunk) == (32, 1)
    adapter.channel_chunk = 3
    draws = []
    set_topology = adapter.channel_model.set_topology
    monkeypatch.setattr(adapter.channel_model, "set_topology",
                        lambda *topology: draws.append(len(topology[0])) or set_topology(*topology))

    targets = [adapter.sample(2, 5.0).channel_target for _ in range(4)]

    assert draws == [6, 6]
    assert not torch.equal(targets[0], targets[1])


def test_explicit_active_users_mute_the_selected_users() -> None:
    adapter = SionnaPuschAdapter.from_config(REPOSITORY_ROOT / "config" / "nrx_rt.cfg", device="cpu")
    active_users = torch.tensor([[0.0, 1.0], [1.0, 0.0]])

    batch = adapter.sample(2, 5.0, num_active_users=1, active_users=active_users)

    torch.testing.assert_close(batch.active_users, active_users)


def test_compiled_decoders_match_eager_decoding() -> None:
    config_path = REPOSITORY_ROOT / "config" / "nrx_rt.cfg"
    cpu = SionnaPuschAdapter.from_config(config_path, mode="evaluation", device="cpu")
    iteration = cpu.decoders[0]._decoder._bp_iter
    cpu.compile_decoders()
    assert cpu.decoders[0]._decoder._bp_iter == iteration
    if not torch.cuda.is_available():
        return

    adapter = SionnaPuschAdapter.from_config(
        config_path, mode="evaluation", device="cuda:0", n_size_bwp_override=4
    )
    llrs = 4 * torch.randn(
        (3, adapter.num_users, 48, 14, adapter.num_bits),
        generator=torch.Generator().manual_seed(0),
    ).cuda()
    eager = adapter.decode(llrs)
    adapter.compile_decoders()
    compiled = adapter.decode(llrs)

    assert all(torch.equal(a, b) for a, b in zip(eager, compiled))


@pytest.mark.parametrize("channel", ["UMa", "AWGN", "DoubleTDLmedium", "DoubleTDLhigh"])
def test_additional_channel_models_generate_batches(tmp_path: Path, channel: str) -> None:
    config_path = tmp_path / "channel.cfg"
    config_path.write_text(
        (REPOSITORY_ROOT / "config" / "nrx_rt.cfg").read_text()
        .replace('channel_type_eval = "DoubleTDLlow"', f'channel_type_eval = "{channel}"')
        .replace("n_size_bwp_eval = 273", "n_size_bwp_eval = 4")
    )
    adapter = SionnaPuschAdapter.from_config(config_path, mode="evaluation", device="cpu")
    batch = adapter.sample(2, 10.0)

    assert adapter.num_users == (2 if channel.startswith("DoubleTDL") else 1)
    assert batch.received.shape == (2, 48, 14, 8)
    assert torch.isfinite(batch.received).all()
    if channel == "AWGN":
        # Flat unit channel; the precoder adds the same real gain everywhere.
        real = batch.channel_target[..., :4]
        torch.testing.assert_close(real, real[0, 0, 0, 0, 0].expand_as(real))
        assert (batch.channel_target[..., 4:] == 0).all()


def test_antenna_correlation_matrix() -> None:
    from neural_rx.phy import _correlation_matrix

    torch.testing.assert_close(_correlation_matrix(4, 0.0), torch.eye(4, dtype=torch.complex64))
    expected = 0.9 ** torch.tensor([0, 1 / 9, 4 / 9, 1])
    torch.testing.assert_close(_correlation_matrix(4, 0.9)[0].real, expected)
    torch.testing.assert_close(_correlation_matrix(4, 0.9)[:, 0].real, expected)


def test_frequency_offset_rotates_each_user(tmp_path: Path) -> None:
    from neural_rx.phy import FrequencyOffset

    adapter = SionnaPuschAdapter.from_config(
        REPOSITORY_ROOT / "config" / "nrx_rt.cfg", device="cpu", n_size_bwp_override=4
    )
    signal, _ = adapter.transmitter(2)
    grid = adapter.resource_grid
    torch.testing.assert_close(
        FrequencyOffset(0.0, grid, True, "cpu")(signal), signal, atol=1e-5, rtol=0
    )
    shifted = FrequencyOffset(1 / grid.fft_size, grid, True, "cpu")(signal)
    # An offset of one subcarrier spacing moves every subcarrier by one bin.
    torch.testing.assert_close(
        shifted[..., 1:].abs(), signal[..., :-1].abs(), atol=1e-4, rtol=0
    )

    config_path = tmp_path / "cfo.cfg"
    config_path.write_text(
        (REPOSITORY_ROOT / "config" / "nrx_rt.cfg").read_text()
        .replace('channel_type = "UMi"', 'channel_type = "UMi"\ncfo_offset_ppm = 0.5')
    )
    adapter = SionnaPuschAdapter.from_config(config_path, device="cpu", n_size_bwp_override=4)
    assert adapter.frequency_offset is not None and not adapter.frequency_offset.constant
    assert torch.isfinite(adapter.sample(2, 10.0).received).all()
