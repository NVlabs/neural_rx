"""Fixed-slot symbol interventions from the pilotless communications tutorial.

Each invocation generates one coded PUSCH slot, replaces excluded data symbols
uniformly from the remaining alphabet, and retransmits that fixed slot over
independent channels. Replacement does not renormalize the transmit power.
Plot returned linear MSE with ``10 * log10(mse)``.
"""

from collections.abc import Sequence

import torch
from torch import Tensor

from neural_rx.phy import SionnaPuschAdapter, _single_sector_topology
from neural_rx.reproducibility import seed_everything


@torch.inference_mode()
def channel_estimation_experiment(
    adapter: SionnaPuschAdapter,
    model: torch.nn.Module,
    *,
    batch_size: int,
    num_trials: int,
    esno_db: float,
    excluded_symbols: Sequence[int],
    seed: int = 42,
    microbatch_size: int = 32,
) -> dict[str, Tensor | int]:
    """Measure channel-readout MSE after actually excluding transmit symbols.

    Use a training-mode four-PRB E2E adapter and its trained receiver/constellation.
    Only user zero transmits, including on adapters
    configured for multiple users. ``batch_size * num_trials`` independent
    channel realizations are evaluated; microbatching bounds activation memory.
    ``esno_db`` always means Es/N0, regardless of the adapter's Eb/N0 setting.

    Returns ``mse_grid`` [1,F,T], scalar ``mean_mse``, the single fixed
    ``transmitted_grid`` [1,U,A,T,F], and integer ``samples``. MSE averages the
    real and imaginary feature errors separately. Repeated calls need distinct
    seeds to generate distinct slots.
    """
    if min(batch_size, num_trials, microbatch_size) < 1:
        raise ValueError("batch_size, num_trials and microbatch_size must be positive")
    points = adapter.constellation_points.detach()
    excluded = sorted(set(int(index) for index in excluded_symbols))
    if any(index < 0 or index >= len(points) for index in excluded):
        raise ValueError("Excluded symbol index is outside the constellation")
    allowed = [index for index in range(len(points)) if index not in excluded]
    if not allowed:
        raise ValueError("At least one constellation symbol must remain")
    if len(adapter.transmitters) != 1:
        raise ValueError("The fixed-slot experiment requires a single configured MCS")
    if adapter.channel_estimator is not None:
        raise ValueError("The pilotless experiment requires initial_chest=None")

    seed_everything(seed)
    if adapter.randomize_pilots:
        pilot_index = torch.randint(len(adapter.pilot_set), (), device=adapter.device)
        adapter.transmitter.pilot_pattern.pilots = adapter.pilot_set[pilot_index]
    slot, _ = adapter.transmitter(1)
    slot = slot.clone()
    slot[:, 1:] = 0
    config = adapter.pusch_configs[0]
    if config.num_layers != 1:
        raise ValueError("The experiment requires one transmit layer")
    if config.precoding_matrix is None:
        precoder = torch.ones(1, dtype=slot.dtype, device=adapter.device)
    else:
        precoder = torch.as_tensor(config.precoding_matrix, device=adapter.device,
                                    dtype=slot.dtype)[:, 0]
    antenna = int(precoder.abs().argmax())
    symbols = slot[0, 0, antenna] / precoder[antenna]
    data_positions = adapter.data_mask[0].T
    allowed_points = points[torch.tensor(allowed, device=adapter.device)]
    for index in excluded:
        positions = data_positions & ((symbols - points[index]).abs() < 1e-6)
        count = int(positions.sum())
        replacements = allowed_points[torch.randint(
            len(allowed), (count,), device=adapter.device
        )]
        slot[0, 0, :, positions] = precoder[:, None] * replacements[None]

    no = torch.tensor(10.0 ** (-esno_db / 10.0), device=adapter.device)
    error_sum = None
    samples = 0
    for _ in range(num_trials):
        for start in range(0, batch_size, microbatch_size):
            count = min(microbatch_size, batch_size - start)
            transmitted = slot.expand(count, -1, -1, -1, -1)
            if adapter.channel_model is not None:
                if adapter.channel_type == "UMi":
                    adapter.channel_model.set_topology(*_single_sector_topology(
                        count, adapter.num_users, adapter.min_ut_velocity,
                        adapter.max_ut_velocity, adapter.device,
                    ))
                if adapter.channel is None:
                    raise RuntimeError("Shared channel was not initialized")
                received, h = adapter.channel(transmitted)
                target = adapter._effective_channel(h, 0, config)
            else:
                # Inactive users transmit exactly zero; their channels need not
                # be generated for this single-user intervention.
                received, h = adapter.channels[0](transmitted[:, :1])
                target = adapter._effective_channel(h, 0, config)
            received = adapter.awgn(received, no)
            received = received[:, 0].permute(0, 3, 2, 1)
            received = torch.cat((received.real, received.imag), dim=-1)
            if adapter.mask_pilots:
                pilot_mask = adapter.pilot_mask.any(dim=0)[None, :, :, None]
                received = received.masked_fill(pilot_mask, 0)
            active = received.new_zeros((count, adapter.num_users))
            active[:, 0] = 1
            mcs_mask = received.new_ones((count, adapter.num_users, 1))
            output = model(received, adapter.pilot_features, active,
                           mcs_mask=mcs_mask)[-1]
            errors = (output.channel[:, 0] - target).square().mean(dim=-1).sum(dim=0)
            error_sum = errors if error_sum is None else error_sum + errors
            samples += count
    mse = (error_sum / samples).unsqueeze(0)
    return {"mse_grid": mse, "mean_mse": mse.mean(),
            "transmitted_grid": slot, "samples": samples}
