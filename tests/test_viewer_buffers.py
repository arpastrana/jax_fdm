import numpy as np
import pytest

from compas.geometry import Cylinder
from compas.geometry import Line
from jax_fdm.visualization.shapes import Arrow
from jax_fdm.visualization.viewers.buffers import arrows_buffer
from jax_fdm.visualization.viewers.buffers import cylinder_template
from jax_fdm.visualization.viewers.buffers import cylinders_buffer
from jax_fdm.visualization.viewers.buffers import rotations_to
from jax_fdm.visualization.viewers.buffers import spheres_buffer


def soup_of(vertices, faces):
    """Expand a compas tessellation into a triangle soup for comparison."""
    return np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64).ravel()]


def sorted_rows(array):
    return np.array(sorted(map(tuple, np.round(array, 6))))


def test_cylinders_buffer_shapes_and_dtype():
    positions, colors = cylinders_buffer(starts=[[0, 0, 0]],
                                         ends=[[0, 0, 1]],
                                         radii=[0.5],
                                         colors=[[1, 0, 0, 1]],
                                         u=16)
    rows = cylinder_template(16).shape[0]

    assert positions.shape == (rows, 3)
    assert colors.shape == (rows, 4)
    assert positions.dtype == np.float32
    assert colors.dtype == np.float32


def test_cylinders_buffer_matches_compas_shape():
    start, end = [1.0, -2.0, 0.5], [3.0, 1.0, 4.0]
    radius = 0.3

    positions, _ = cylinders_buffer([start], [end], [radius], [[0, 0, 1, 1]], u=16)

    cylinder = Cylinder.from_line_and_radius(Line(start, end), radius)
    expected = soup_of(*cylinder.to_vertices_and_faces(u=16, triangulated=True))

    assert np.allclose(sorted_rows(positions), sorted_rows(expected), atol=1e-5)


def test_rotations_degenerate_directions():
    directions = np.array([[0.0, 0.0, 1.0],
                           [0.0, 0.0, -1.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]])
    rotations = rotations_to(directions)

    for rotation, direction in zip(rotations, directions):
        assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
        assert np.allclose(rotation @ [0.0, 0.0, 1.0], direction, atol=1e-12)


def test_arrows_buffer_matches_arrow_shape():
    anchor, vector = [0.5, 1.5, -1.0], [1.0, 2.0, 0.5]

    positions, _ = arrows_buffer([anchor], [vector], [[0, 1, 0, 1]], u=8)

    arrow = Arrow(position=anchor, direction=vector,
                  head_portion=0.12, head_width=0.04, body_width=0.012)
    expected = soup_of(*arrow.to_vertices_and_faces(u=8, triangulated=True))

    assert np.allclose(sorted_rows(positions), sorted_rows(expected), atol=1e-5)


def test_arrows_buffer_degenerate_collapses_at_anchor():
    anchor = [2.0, 3.0, 4.0]
    live, _ = arrows_buffer([anchor], [[0.0, 0.0, 1.0]], [[0, 0, 0, 1]])
    degenerate, _ = arrows_buffer([anchor], [[0.0, 0.0, 0.0]], [[0, 0, 0, 1]])

    # constant topology: same soup size whether the arrow is live or collapsed
    assert degenerate.shape == live.shape
    assert np.allclose(degenerate, np.array(anchor, dtype=np.float32))


def test_spheres_buffer_center_and_radius():
    centers = [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]
    radii = [1.0, 0.25]

    positions, _ = spheres_buffer(centers, radii, [[0, 0, 0, 1]] * 2)
    per_sphere = positions.reshape(2, -1, 3)

    for soup, center, radius in zip(per_sphere, centers, radii):
        distances = np.linalg.norm(soup - np.array(center, dtype=np.float32), axis=1)
        assert np.max(distances) == pytest.approx(radius, rel=1e-5)


def test_colors_repeat_per_element():
    colors = [[1, 0, 0, 1], [0, 1, 0, 0.5]]
    positions, facecolors = cylinders_buffer(starts=[[0, 0, 0], [5, 0, 0]],
                                             ends=[[0, 0, 1], [5, 0, 1]],
                                             radii=[0.1, 0.1],
                                             colors=colors)

    assert facecolors.shape[0] == positions.shape[0]
    per_element = facecolors.reshape(2, -1, 4)
    for rows, color in zip(per_element, colors):
        assert np.allclose(rows, np.array(color, dtype=np.float32))


def test_empty_inputs():
    for builder, kwargs in ((cylinders_buffer, dict(starts=[], ends=[], radii=[], colors=[])),
                            (arrows_buffer, dict(anchors=[], vectors=[], colors=[])),
                            (spheres_buffer, dict(centers=[], radii=[], colors=[]))):
        positions, colors = builder(**kwargs)
        assert positions.shape == (0, 3)
        assert colors.shape == (0, 4)
