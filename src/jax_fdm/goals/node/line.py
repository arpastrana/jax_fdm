import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.geometry import closest_point_on_line
from jax_fdm.goals.node import NodePointGoal


class NodeLineGoal(NodePointGoal):
    """
    Pulls the position of a node to a target line ray.
    """
    @property
    def target(self) -> Float[Array, "elements 2 3"]:
        """
        The target to achieve
        """
        return self._target

    @target.setter
    def target(self, target: Float[Array, "..."]) -> None:
        self._target = jnp.reshape(jnp.asarray(target), (-1, 2, 3))

    def goal(self, target: Float[Array, "2 3"], prediction: Float[Array, "3"]) -> Float[Array, "3"]:
        """
        The closest point on the target line.
        """
        return closest_point_on_line(prediction, target)
