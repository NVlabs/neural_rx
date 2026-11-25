# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Sionna RT scene construction and CIR dataset generation."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from neural_rx.cir_dataset import CIRDataset


def sample_trajectory(
    waypoints: Sequence[Sequence[float]] | Tensor,
    num_points: int,
    speed: float,
) -> tuple[Tensor, Tensor, float]:
    """Sample equidistant positions and constant-speed segment velocities."""

    points = torch.as_tensor(waypoints, dtype=torch.float32)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 3:
        raise ValueError("waypoints must have shape [at least two, 3]")
    if num_points < 1:
        raise ValueError("num_points must be positive")
    if not math.isfinite(speed) or speed < 0:
        raise ValueError("speed must be finite and non-negative")

    segments = points[1:] - points[:-1]
    lengths = torch.linalg.vector_norm(segments, dim=1)
    if torch.any(lengths == 0):
        raise ValueError("Consecutive trajectory waypoints must be distinct")
    total_distance_tensor = lengths.sum()
    sample_distance = total_distance_tensor / num_points
    distances = torch.arange(num_points, dtype=points.dtype) * sample_distance
    cumulative = torch.cumsum(lengths, dim=0)
    segment_indices = torch.bucketize(distances, cumulative, right=True)
    segment_starts = torch.cat([lengths.new_zeros(1), cumulative[:-1]])
    directions = segments / lengths[:, None]
    offsets = distances - segment_starts[segment_indices]
    positions = points[segment_indices] + directions[segment_indices] * offsets[:, None]
    velocities = directions[segment_indices] * speed
    return positions, velocities, float(total_distance_tensor)


def build_munich_scene(
    *,
    carrier_frequency: float = 2.14e9,
    num_bs_antennas: int = 4,
    num_ut_antennas: int = 2,
):
    """Construct the Munich scene and isotropic arrays used by the site experiment."""

    if num_bs_antennas < 2 or num_bs_antennas % 2:
        raise ValueError("num_bs_antennas must be a positive even number")
    if num_ut_antennas < 1:
        raise ValueError("num_ut_antennas must be positive")

    from sionna.rt import PlanarArray, Transmitter, load_scene, scene as scene_files

    scene = load_scene(scene_files.munich)
    scene.frequency = carrier_frequency
    scene.tx_array = PlanarArray(
        num_rows=num_bs_antennas // 2,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="cross",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=num_ut_antennas,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.add(
        Transmitter(
            name="tx",
            position=[-210.0, 50.0, 106.0],
            orientation=[0.0, 0.0, 0.0],
            color=(1.0, 0.0, 0.0),
        )
    )
    return scene


def _fixed_path_count(coefficients: Tensor, delays: Tensor, num_paths: int) -> tuple[Tensor, Tensor]:
    current_paths = coefficients.shape[-2]
    if current_paths > num_paths:
        # Match Sionna 0.19 Paths.pad_or_crop: rank first-time-step
        # amplitudes per antenna link, not the solver's arbitrary path order.
        indices = coefficients[..., 0].abs().topk(num_paths, dim=-1).indices
        selected = coefficients.gather(-2, indices[..., None].expand(
            *indices.shape, coefficients.shape[-1]))
        delay_indices = indices[:, :, 0, :, 0]
        return selected, delays.gather(-1, delay_indices)
    if current_paths == num_paths:
        return coefficients, delays
    coefficient_padding = coefficients.new_zeros(
        (*coefficients.shape[:-2], num_paths - current_paths, coefficients.shape[-1])
    )
    delay_padding = delays.new_full((*delays.shape[:-1], num_paths - current_paths), -1)
    return (
        torch.cat([coefficients, coefficient_padding], dim=-2),
        torch.cat([delays, delay_padding], dim=-1),
    )


def generate_rt_cir_dataset(
    scene,
    positions: Sequence[Tensor],
    velocities: Sequence[Tensor],
    *,
    batch_size: int,
    num_paths: int,
    num_time_steps: int,
    sampling_frequency: float,
    samples_per_src: int,
    max_depth: int = 5,
    refraction: bool = True,
    target_num_samples: int | None = None,
    seed: int = 42,
    progress_callback=None,
) -> CIRDataset:
    """Trace positions with ``PathSolver`` and return non-empty uplink CIRs."""

    if len(positions) == 0 or len(positions) != len(velocities):
        raise ValueError("positions and velocities must define the same non-empty user list")
    position_tensors = [torch.as_tensor(item, dtype=torch.float32) for item in positions]
    velocity_tensors = [torch.as_tensor(item, dtype=torch.float32) for item in velocities]
    num_candidates = position_tensors[0].shape[0]
    for user_positions, user_velocities in zip(position_tensors, velocity_tensors):
        if user_positions.shape != (num_candidates, 3):
            raise ValueError("Every position tensor must have shape [samples, 3]")
        if user_velocities.shape != user_positions.shape:
            raise ValueError("Velocity tensors must match their position tensors")
    if batch_size < 1 or num_paths < 1 or num_time_steps < 1 or samples_per_src < 1:
        raise ValueError("Ray-tracing dimensions and sample counts must be positive")
    if sampling_frequency <= 0 or not math.isfinite(sampling_frequency):
        raise ValueError("sampling_frequency must be positive and finite")
    if target_num_samples is not None and target_num_samples < 1:
        raise ValueError("target_num_samples must be positive")

    from sionna.rt import PathSolver, Receiver

    saved_coefficients = []
    saved_delays = []
    for start in range(0, num_candidates, batch_size):
        stop = min(start + batch_size, num_candidates)
        user_coefficients = []
        user_delays = []
        for user, (user_positions, user_velocities) in enumerate(
            zip(position_tensors, velocity_tensors)
        ):
            receiver_names = []
            for local_index, sample_index in enumerate(range(start, stop)):
                name = f"dataset-rx-{user}-{local_index}"
                receiver_names.append(name)
                scene.add(
                    Receiver(
                        name=name,
                        position=user_positions[sample_index].tolist(),
                        velocity=user_velocities[sample_index].tolist(),
                    )
                )
            try:
                paths = PathSolver()(
                    scene,
                    max_depth=max_depth,
                    samples_per_src=samples_per_src,
                    synthetic_array=True,
                    los=True,
                    specular_reflection=True,
                    diffuse_reflection=False,
                    refraction=refraction,
                    diffraction=True,
                    seed=seed + start + user,
                )
                coefficients, delays = paths.cir(
                    sampling_frequency=sampling_frequency,
                    num_time_steps=num_time_steps,
                    normalize_delays=True,
                    reverse_direction=True,
                    out_type="torch",
                )
            finally:
                for name in receiver_names:
                    scene.remove(name)

            coefficients = coefficients.permute(2, 0, 1, 3, 4, 5).unsqueeze(3)
            delays = delays.permute(1, 0, 2).unsqueeze(2)
            coefficients, delays = _fixed_path_count(coefficients, delays, num_paths)
            user_coefficients.append(coefficients.to(torch.complex64).cpu())
            user_delays.append(delays.to(torch.float32).cpu())

        batch_coefficients = torch.cat(user_coefficients, dim=3)
        batch_delays = torch.cat(user_delays, dim=2)
        path_power = batch_coefficients.abs().sum(dim=(1, 2, 4, 5, 6))
        valid = torch.all(path_power > 0, dim=1)
        saved_coefficients.append(batch_coefficients[valid])
        saved_delays.append(batch_delays[valid])
        if progress_callback is not None:
            progress_callback(start, stop, int(valid.sum()))
        if target_num_samples is not None:
            count = sum(item.shape[0] for item in saved_coefficients)
            if count >= target_num_samples:
                break

    coefficients = torch.cat(saved_coefficients, dim=0)
    delays = torch.cat(saved_delays, dim=0)
    if target_num_samples is not None:
        if coefficients.shape[0] < target_num_samples:
            raise RuntimeError(
                f"Ray tracing produced {coefficients.shape[0]} valid samples; "
                f"{target_num_samples} were requested"
            )
        coefficients = coefficients[:target_num_samples]
        delays = delays[:target_num_samples]
    return CIRDataset(coefficients=coefficients, delays=delays)
