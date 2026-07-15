import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.geometry import closest_point_on_plane
from jax_fdm.goals.node import NodePointGoal


class NodePlaneGoal(NodePointGoal):
    """
    Pulls the xyz position of a node to a target plane.
    """
    @property
    def target(self) -> Float[Array, "elements 2 3"] | None:
        """
        The target to achieve.
        """
        return self._target

    @target.setter
    def target(self, target: Float[Array, "..."]) -> None:
        self._target = jnp.reshape(jnp.asarray(target), (-1, 2, 3))

    @staticmethod
    def goal(target: Float[Array, "2 3"], prediction: Float[Array, "3"]) -> Float[Array, "3"]:
        """
        Calculate the closest point on the target plane given the current node coordinates.
        """
        return closest_point_on_plane(prediction, target)
