import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.geometry import closest_point_on_plane
from jax_fdm.goals.goal import TargetLike
from jax_fdm.goals.node import NodePointGoal

__all__ = ["NodePlaneGoal"]


class NodePlaneGoal(NodePointGoal):
    """
    Pull a node onto a target plane, defined by a point and a normal.
    """

    @property
    def target(self) -> Float[Array, "elements 2 3"]:
        """
        The origin and normal defining the target plane of each element.
        """
        return self._target

    @target.setter
    def target(self, target: TargetLike) -> None:
        self._target = jnp.reshape(jnp.asarray(target), (-1, 2, 3))

    def goal(
        self,
        target: Float[Array, "2 3"],
        prediction: Float[Array, "3"],
    ) -> Float[Array, "3"]:
        """
        The point on the target plane closest to the node.

        Parameters
        ----------
        target :
            The origin and normal defining the target plane.
        prediction :
            The current node coordinates.

        Returns
        -------
        goal :
            The orthogonal projection of the node onto the plane.
        """
        return closest_point_on_plane(prediction, target)
