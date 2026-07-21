import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.geometry import closest_point_on_line
from jax_fdm.goals.goal import TargetLike
from jax_fdm.goals.node.point import NodePointGoal

__all__ = ["NodeLineGoal"]


class NodeLineGoal(NodePointGoal):
    """
    Pull a node onto a target line, defined by two points.
    """

    @property
    def target(self) -> Float[Array, "elements 2 3"]:
        """
        The two points defining the target line of each element.
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
        The point on the target line closest to the node.

        Parameters
        ----------
        target :
            The two points defining the target line.
        prediction :
            The current node coordinates.

        Returns
        -------
        goal :
            The closest point on the (infinite) line through the two points.
        """
        return closest_point_on_line(prediction, target)
