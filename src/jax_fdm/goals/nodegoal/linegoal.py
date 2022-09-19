import jax.numpy as jnp

from compas.geometry import closest_point_on_line

from jax_fdm.goals import NodesPointGoal


class NodesLineGoal(NodesPointGoal):
    """
    Pulls the xyz position of a node to a target line ray.
    """
    def __init__(self, keys, targets, weights=1.0):
        super().__init__(keys=keys, targets=targets, weights=weights)

    def target(self, prediction):
        """
        """
        line = self._target
        point = closest_point_on_line(prediction, line)

        return jnp.array(point, dtype=jnp.float64)

    @staticmethod
    def _closest_point_on_line(point, line):
        """
        Computes the closest location on a line to a supplied point.
        """
        pass
