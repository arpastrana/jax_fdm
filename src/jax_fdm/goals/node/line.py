import numpy as np

from jax_fdm.geometry import closest_point_on_line

from jax_fdm.goals.node import NodePointGoal


class NodeLineGoal(NodePointGoal):
    """
    Pulls the xyz position of a node to a target line ray.
    """
    @property
    def target(self):
        """
        The target to achieve
        """
        return self._target

    @target.setter
    def target(self, target):
        self._target = np.reshape(target, (-1, 2, 3))

    @staticmethod
    def goal(target, prediction):
        """
        """
        return closest_point_on_line(prediction, target)


if __name__ == "__main__":

    import jax.numpy as jnp

    from jax import vmap

    from compas.geometry import Line

    points = [[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.5, 0.0]]
    points = jnp.array(points)

    line = Line([1.0, 1.0, 0.0], [1.0, -1.0, 0.0])
    lines = [line] * len(points)
    lines = np.reshape(lines, (-1, 2, 3))

    closest = vmap(closest_point_on_line)(points, lines)
    print(closest)
    print(closest.shape)

    assert np.allclose(closest, jnp.array([[1.0, 0.0, 0.0], [1.0, -0.5, 0.0], [1.0, 0.5, 0.0]])), closest
    print("Done!")
