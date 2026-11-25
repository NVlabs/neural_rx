"""Native Sionna RT figures for the Munich site experiment."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from raytracing.scene import sample_trajectory


EVALUATION_WAYPOINTS = (
    ((520, 65, 1.5), (125, 175, 1.5), (90, 30, 1.5), (85, 0, 1.5),
     (30, -115, 1.5), (110, -175, 1.5), (200, -210, 1.5),
     (245, -220, 1.5), (400, -285, 1.5)),
    ((53, 310, 1.5), (-72, -91, 1.5), (-380, 40, 1.5),
     (-485, 50, 1.5), (-650, 95, 1.5)),
)


def evaluation_trajectories(num_points: int = 10_000):
    """Return position, velocity, and length tuples for both users."""
    return tuple(sample_trajectory(points, num_points, speed)
                 for points, speed in zip(EVALUATION_WAYPOINTS, (3.5, 3.0)))


def build_site_radio_map(scene, *, samples_per_tx: int = 10_000_000,
                         max_depth: int = 6, cell_size: float = 2.0,
                         seed: int = 42):
    """Compute the path-gain radio map of ``scene``."""
    from sionna.rt import RadioMapSolver

    return RadioMapSolver()(
        scene, cell_size=(cell_size, cell_size), samples_per_tx=samples_per_tx,
        max_depth=max_depth, los=True, specular_reflection=True,
        diffuse_reflection=False, refraction=False, diffraction=True, seed=seed)


def sample_training_positions(radio_map, num_positions: int = 1_200,
                              *, seed: int = 42) -> torch.Tensor:
    """Sample covered cell areas with ``center_pos=False``.

    The two cell-edge vectors preserve the measurement plane's orientation.
    A separate random stream avoids coupling offsets to notebook velocities.
    """
    if num_positions < 1:
        raise ValueError("num_positions must be positive")
    cells, counts = radio_map.sample_cells(
        num_positions, metric="path_gain", min_val_db=-130., max_val_db=0.,
        min_dist=20., max_dist=700., seed=seed,
    )
    count = int(counts.numpy()[0])
    if count == 0:
        raise RuntimeError("No covered cells satisfy the training-position bounds")
    indices = torch.as_tensor(cells.numpy()[0, :count], dtype=torch.long)
    centers = torch.as_tensor(radio_map.cell_centers.numpy(), dtype=torch.float32)
    if centers.ndim != 3 or min(centers.shape[:2]) < 2:
        raise ValueError("Sampling cell areas requires at least a 2-by-2 planar radio map")
    positions = centers.reshape(-1, 3)[indices].clone()
    half_row = 0.5 * (centers[0, 0] - centers[1, 0])
    half_column = 0.5 * (centers[0, 0] - centers[0, 1])
    generator = torch.Generator().manual_seed((seed + 1_000_003) % (2**63))
    row_offset = 2 * torch.rand((count, 1), generator=generator) - 1
    column_offset = 2 * torch.rand((count, 1), generator=generator) - 1
    positions += row_offset * half_row + column_offset * half_column
    return positions


def render_positions(scene, radio_map, positions: Sequence[torch.Tensor], *,
                     training: bool = False, stride: int = 10,
                     resolution: tuple[int, int] = (700, 500),
                     num_samples: int = 128):
    """Render camera views, restoring all temporary receivers afterward.

    Pass both trajectories for the evaluation view. For the training view pass
    the actual last CIR-generation batch and ``stride=1``.
    """
    if stride < 1:
        raise ValueError("stride must be positive")
    from sionna.rt import Camera, Receiver

    names = []
    try:
        for user, points in enumerate(positions):
            for index, point in enumerate(points[::stride]):
                name = f"notebook-visualization-{user}-{index}"
                if scene.get(name) is not None:
                    raise ValueError(f"Scene already contains {name}")
                scene.add(Receiver(name=name, position=point.tolist(),
                                   color=(1.0, 0.5, 0.5)))
                names.append(name)
        camera = Camera(position=[-1200. if training else -1500., 100., 1200.])
        camera.look_at(scene.get("tx"))
        return scene.render(camera=camera, resolution=resolution,
                            radio_map=radio_map, num_samples=num_samples,
                            rm_show_color_bar=True)
    finally:
        for name in names:
            scene.remove(name)


def coverage_figure(radio_map):
    """Return the native path-gain coverage-map figure."""
    return radio_map.show()
