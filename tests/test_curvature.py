import jax.numpy as jnp

from jax_fdm.geometry import curvature_points


def test_curvature_colinear_2d():
    """
    Colinear points in 2D yield zero energy.
    """
    points = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    energy = curvature_points(points)
    assert jnp.allclose(energy, 0.0), f"Colinear 2D: {energy}"


def test_curvature_colinear_3d():
    """
    Colinear points in 3D yield zero energy.
    """
    points = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    energy = curvature_points(points)
    assert jnp.allclose(energy, 0.0), f"Colinear 3D: {energy}"


def test_curvature_non_colinear():
    """
    Non-colinear points (L-shape) yield positive energy.
    """
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    energy = curvature_points(points)
    assert energy > 0.0, f"L-shape: {energy}"


def test_curvature_scale_invariant():
    """
    Curvature is scale-invariant (depends only on turn angles).
    """
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    energy = curvature_points(points)
    points_scaled = points * 10.0
    energy_scaled = curvature_points(points_scaled)
    assert jnp.allclose(energy, energy_scaled), f"{energy} vs {energy_scaled}"


def test_curvature_invariant_to_problem_size():
    """
    Curvature is invariant to problem size (normalized by number of interior vertices).
    Same L-shape with more points (subdivided) yields same or lower energy, not higher.
    """
    points_3 = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    points_5 = jnp.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    energy_3 = curvature_points(points_3)
    energy_5 = curvature_points(points_5)
    assert energy_5 <= energy_3, f"Subdividing should not increase energy: {energy_5} vs {energy_3}"
