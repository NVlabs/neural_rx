import torch

from raytracing.scene import _fixed_path_count, sample_trajectory


def test_sample_trajectory_is_equidistant_with_segment_velocities() -> None:
    positions, velocities, total_distance = sample_trajectory(
        [[0.0, 0.0, 1.5], [3.0, 0.0, 1.5], [3.0, 4.0, 1.5]],
        num_points=7,
        speed=2.5,
    )

    expected_positions = torch.tensor(
        [
            [0.0, 0.0, 1.5],
            [1.0, 0.0, 1.5],
            [2.0, 0.0, 1.5],
            [3.0, 0.0, 1.5],
            [3.0, 1.0, 1.5],
            [3.0, 2.0, 1.5],
            [3.0, 3.0, 1.5],
        ]
    )
    expected_velocities = torch.tensor(
        [
            [2.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 2.5, 0.0],
            [0.0, 2.5, 0.0],
        ]
    )
    torch.testing.assert_close(positions, expected_positions)
    torch.testing.assert_close(velocities, expected_velocities)
    assert total_distance == 7.0


def test_fixed_path_count_retains_strongest_paths_and_matching_delays():
    coefficients = torch.tensor([1., 9., 3., 7.], dtype=torch.complex64).reshape(1, 1, 1, 1, 1, 4, 1)
    coefficients = coefficients.expand(1, 1, 4, 1, 2, 4, 14)
    delays = torch.tensor([10., 20., 30., 40.]).reshape(1, 1, 1, 4)
    a, tau = _fixed_path_count(coefficients, delays, 2)
    torch.testing.assert_close(a[..., 0], torch.tensor([9., 7.], dtype=torch.complex64).expand(1, 1, 4, 1, 2, 2))
    torch.testing.assert_close(tau, torch.tensor([20., 40.]).reshape(1, 1, 1, 2))
