import jax.numpy as jnp

from compas.geometry import closest_point_on_line
from compas.geometry import closest_point_on_plane

from dfdm.goals import VectorGoal

from dfdm.goals.nodegoal import NodeGoal


class PointGoal(VectorGoal, NodeGoal):
    """
    Make a node of a network to reach target xyz coordinates.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def prediction(self, eq_state, index):
        """
        The current xyz coordinates of the node in a network.
        """
        return eq_state.xyz[index, :]

    def target(self, prediction):
        """
        The target xyz coordinates.
        """
        return jnp.array(self._target, dtype=jnp.float64)


class LineGoal(PointGoal):
    """
    Pulls the xyz position of a node to a target line ray.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def target(self, prediction):
        """
        """
        line = self._target
        point = closest_point_on_line(prediction, line)

        return jnp.array(point, dtype=jnp.float64)


class PlaneGoal(PointGoal):
    """
    Pulls the xyz position of a node to a target plane.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def target(self, prediction):
        """
        """
        point = prediction
        plane = self._target
        point = closest_point_on_plane(point, plane)

        return jnp.array(point, dtype=jnp.float64)
