from functools import partial

import numpy as np

import jax.numpy as jnp

from jax import vmap

from jax import jit

from jax_fdm.goals.nodegoal import NodesPointGoal


class NodesLineGoal(NodesPointGoal):
    """
    Pulls the xyz position of a node to a target line ray.
    """
    def __init__(self, keys, targets, weights=1.0):
        # match keys, targets and weights
        num_keys = len(keys)
        assert num_keys >= 1, "The list of input keys cannot be empty!"
        super().__init__(keys=keys, targets=targets, weights=weights)

    def target(self, prediction):
        """
        """
        lines = self._target
        start = [line[0] for line in lines]
        start = np.reshape(np.array(start), (-1, 3))
        end = [line[1] for line in lines]
        end = np.reshape(np.array(end), (-1, 3))

        return vmap(closest_point_on_line)(prediction, start, end)


def closest_point_on_line(point, start, end):
    """
    Computes the closest location on a line to a supplied point.
    """
    a, b = start, end
    ab = b - a
    ap = point - a
    c = vector_projection(ap, ab)
    return a + c


def vector_projection(u, v):
    """
    Calculates the orthogonal projection of u onto v.
    """
    l2 = np.sum(v ** 2)
    x = (u @ np.transpose(v)) / l2
    return v * x


if __name__ == "__main__":
    from compas.geometry import Line

    # point = [0.0, 0.0, 0.0]
    # line = ([1.0, 1.0, 0.0], [1.0, -1.0, 0.0])

    # point = jnp.array(point)
    # line = [jnp.array(x) for x in line]

    # closest = closest_point_on_line(point, line)
    # assert np.allclose(closest, [1.0, 0.0, 0.0]), closest
    # print("Done!")

    points = [[0.0, 0.0, 0.0]]  # , [0.0, -0.5, 0.0], [0.0, 0.5, 0.0]]
    line = Line([1.0, 1.0, 0.0], [1.0, -1.0, 0.0])

    lines = [line] * len(points)

    points = jnp.array(points)

    # convert lines into arrays
    starts = []
    ends = []
    for start, end in lines:
        starts.append(start)
        ends.append(end)

    starts = jnp.reshape(jnp.array(starts), (-1, 3))
    ends = jnp.reshape(jnp.array(ends), (-1, 3))

    print(points.shape, starts.shape, ends.shape)
    # targets = []
    # for element in lines:
    #     target_array = jnp.reshape(jnp.array(element), (-1, 3))
    #     targets.append(target_array)

    closest = vmap(closest_point_on_line)(points, starts, ends)
    print(closest)
    print(closest.shape)

    # assert np.allclose(closest, jnp.array([[1.0, 0.0, 0.0], [1.0, -0.5, 0.0], []])), closest
    print("Done!")
