import jax.numpy as jnp

from compas.geometry import closest_point_on_line
from compas.geometry import closest_point_on_plane

from jax_fdm.goals import VectorGoal

from jax_fdm.goals.nodegoal import NodeGoal


class NodePointGoal(VectorGoal, NodeGoal):
    """
    Make a node of a network to reach target xyz coordinates.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def prediction(self, eq_state):
        """
        The current xyz coordinates of the node in a network.
        """
        return eq_state.xyz[self.index, :]


class NodeLineGoal(NodePointGoal):
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


class NodePlaneGoal(NodePointGoal):
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
        point = closest_point_on_plane(prediction, plane)

        return jnp.array(point, dtype=jnp.float64)
