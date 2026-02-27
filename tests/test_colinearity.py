import jax.numpy as jnp

from jax_fdm.geometry import colinearity_points


def test_colinearity_colinear_2d():
    """
    Colinear points in 2D yield zero energy.
    """
    points = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    energy = colinearity_points(points)
    assert jnp.allclose(energy, 0.0), f"Colinear 2D: {energy}"


def test_colinearity_colinear_3d():
    """
    Colinear points in 3D yield zero energy.
    """
    points = jnp.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    energy = colinearity_points(points)
    assert jnp.allclose(energy, 0.0), f"Colinear 3D: {energy}"


def test_colinearity_non_colinear():
    """
    Non-colinear points (L-shape) yield positive energy.
    """
    points = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    energy = colinearity_points(points)
    assert energy > 0.0, f"L-shape: {energy}"


def test_colinearity_scale_invariant():
    """
    Colinearity is scale-invariant for colinear point sequences.
    """
    points = jnp.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    energy = colinearity_points(points)
    points_scaled = points * 10.0
    energy_scaled = colinearity_points(points_scaled)
    assert jnp.allclose(energy, energy_scaled), f"{energy} vs {energy_scaled}"


def test_colinearity_invariant_to_uneven_spacing():
    """
    Colinearity is invariant to uneven spacing for same turn angles.
    Same L-shape with different leg lengths yields the same energy.
    """
    points_uniform = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    points_uneven = jnp.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0]])
    energy_uniform = colinearity_points(points_uniform)
    energy_uneven = colinearity_points(points_uneven)
    assert jnp.allclose(energy_uniform, energy_uneven), f"{energy_uniform} vs {energy_uneven}"


def test_colinearity_invariant_to_uneven_spacing_zigzag():
    """
    Colinearity is invariant to uneven edge lengths for same turn sequence.
    Same zigzag (two 90-degree turns) with uniform vs uneven spacing.
    """
    points_uniform = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    points_uneven = jnp.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    energy_uniform = colinearity_points(points_uniform)
    energy_uneven = colinearity_points(points_uneven)
    assert jnp.allclose(energy_uniform, energy_uneven), f"{energy_uniform} vs {energy_uneven}"


def test_colinearity_invariant_to_problem_size():
    """
    Colinearity is invariant to problem size (normalized by number of interior vertices).
    Same L-shape with more points (subdivided) yields same or lower energy, not higher.
    """
    points_3 = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    points_5 = jnp.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]])
    energy_3 = colinearity_points(points_3)
    energy_5 = colinearity_points(points_5)
    assert energy_5 <= energy_3, f"Subdividing should not increase energy: {energy_5} vs {energy_3}"
