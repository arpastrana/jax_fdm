import numpy as np

import jax.numpy as jnp

from jax_fdm.goals.nodegoal import NodesPointGoal


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
        return self._closest_point_on_plane(points=prediction, planes=self._target)

    @staticmethod
    def _closest_point_on_plane(points, planes):
        bases, normals = planes
        normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)
        d = jnp.einsum('ij,ij->i', normals, bases)
        e = jnp.einsum('ij,ij->i', normals, points) - d
        k = jnp.reshape(e / jnp.sum(jnp.square(normals), axis=-1), (-1, 1))
        return points - normals * k
