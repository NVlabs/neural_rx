"""Check geometry and masked radio-map samples without launching ray tracing."""

import numpy as np
import pytest
import torch

from raytracing.visualization import evaluation_trajectories, sample_training_positions


class Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def numpy(self):
        return self.values


class Map:
    cell_centers = Array([[[0, 0, 1.5], [2, 0, 1.5], [4, 0, 1.5]],
                         [[0, 2, 1.5], [2, 2, 1.5], [4, 2, 1.5]]])

    def sample_cells(self, count, **kwargs):
        assert count == 3
        assert set(kwargs) == {"metric", "min_val_db", "max_val_db", "min_dist", "max_dist", "seed"}
        assert {k: v for k, v in kwargs.items() if k != "seed"} == dict(
            metric="path_gain", min_val_db=-130., max_val_db=0., min_dist=20., max_dist=700.)
        return Array([[2, 1, 99999]]), Array([2])


def test_sampling_ignores_undefined_cells_and_uses_site_bounds():
    positions = sample_training_positions(Map(), 3, seed=7)
    centers = torch.tensor([[4., 0., 1.5], [2., 0., 1.5]])
    offsets = positions - centers
    assert positions.shape == (2, 3)
    assert (offsets[:, :2].abs() <= 1.).all()
    assert (offsets[:, :2].abs() > 0.).all()
    torch.testing.assert_close(positions[:, 2], torch.full((2,), 1.5))
    torch.testing.assert_close(positions, sample_training_positions(Map(), 3, seed=7))
    assert not torch.equal(positions, sample_training_positions(Map(), 3, seed=8))


def test_sampling_offsets_follow_rotated_measurement_plane():
    rotated = Map()
    # Rotate XY into XZ; the map's two-metre cell edges remain orthogonal.
    centers = np.asarray(Map.cell_centers.numpy()).copy()
    centers = centers[..., [0, 2, 1]]
    rotated.cell_centers = Array(centers)
    positions = sample_training_positions(rotated, 3, seed=7)
    reference = sample_training_positions(Map(), 3, seed=7)[:, [0, 2, 1]]
    torch.testing.assert_close(positions, reference)


def test_trajectories_have_correct_geometry_and_speed():
    trajectories = evaluation_trajectories(100)
    for (positions, velocities, length), start, speed in zip(
        trajectories, ((520., 65., 1.5), (53., 310., 1.5)), (3.5, 3.0)
    ):
        assert positions.shape == velocities.shape == (100, 3)
        torch.testing.assert_close(positions[0], torch.tensor(start))
        torch.testing.assert_close(velocities.norm(dim=-1), torch.full((100,), speed))
        assert length > 1000


def test_invalid_sample_count():
    with pytest.raises(ValueError, match="positive"):
        sample_training_positions(Map(), 0)


@pytest.mark.parametrize("training", [False, True])
def test_render_positions_preserves_quantitative_colorbar(monkeypatch, training):
    import sys
    import types
    from raytracing.visualization import render_positions

    class Camera:
        def __init__(self, **kwargs):
            pass

        def look_at(self, target):
            pass

    class Scene:
        def get(self, name):
            return object()

        def render(self, **kwargs):
            assert kwargs["radio_map"] is radio_map
            assert kwargs.get("rm_show_color_bar") is True
            return "figure"

    monkeypatch.setitem(sys.modules, "sionna.rt", types.SimpleNamespace(Camera=Camera, Receiver=object))
    radio_map = object()
    assert render_positions(Scene(), radio_map, [torch.empty(0, 3)], training=training) == "figure"
