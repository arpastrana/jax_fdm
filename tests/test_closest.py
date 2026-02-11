import numpy as np
import jax.numpy as jnp

from compas.geometry import closest_point_on_line as compas_closest_point_on_line
from jax_fdm.geometry import closest_point_on_line


def test_closest_point_on_line():
    """
    Test the closest point on a line.
    """
    for _ in range(10):
        point = jnp.array(np.random.rand(3))
        line = [np.random.rand(3), np.random.rand(3)]
        closest_compas = compas_closest_point_on_line(point, line)
        closest = closest_point_on_line(jnp.array(point), jnp.array(line))

        assert np.allclose(closest, closest_compas), closest
