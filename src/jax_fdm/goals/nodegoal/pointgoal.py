import numpy as np

import jax.numpy as jnp

from compas.geometry import closest_point_on_line

from jax_fdm.goals import VectorGoal

from jax_fdm.goals.nodegoal import NodesGoal


class NodesPointGoal(VectorGoal, NodesGoal):
    """
    Make a node of a network to reach target xyz coordinates.
    """
    def __init__(self, keys, targets, weights=1.0):
        super().__init__(keys=keys, targets=targets, weights=weights)

    def prediction(self, eq_state):
        """
        The current xyz coordinates of the node in a network.
        """
        return eq_state.xyz[self.index, :]


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


class NodesPlaneGoal(NodesPointGoal):
    """
    Pulls the xyz position of a node to a target plane.
    """
    def __init__(self, keys, targets, weights=1.0):
        origins = np.asarray([x[0] for x in targets])
        planes = np.asarray([x[1] for x in targets])
        targets = (origins, planes)

        super().__init__(keys=keys, targets=targets, weights=weights)

    def target(self, prediction):
        """
        """
        points = prediction
        planes = self._target
        points = self._closest_points_on_planes(points, planes)
        return points

    @staticmethod
    def _closest_points_on_planes(points, planes):
        bases, normals = planes
        normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
        d = jnp.einsum('ij,ij->i', normals, bases)
        e = jnp.einsum('ij,ij->i', normals, points) - d
        k = jnp.reshape(e / jnp.sum(jnp.square(normals), axis=-1), (-1, 1))
        return points - normals * k
