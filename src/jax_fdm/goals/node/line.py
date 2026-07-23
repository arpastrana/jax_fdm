from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.geometry import closest_point_on_line
from jax_fdm.goals.node.point import NodePointGoal

__all__ = ["NodeLineGoal"]


class NodeLineGoal(NodePointGoal):
    """
    Pull a node onto a target line, defined by two points.

    Notes
    -----
    The target is the two points of the line, one ``(2, 3)`` array per element;
    `tree_stack` adds the leading element axis when goals are grouped.
    """

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
