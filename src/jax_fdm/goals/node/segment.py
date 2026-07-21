from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.geometry import closest_point_on_segment
from jax_fdm.goals.node.line import NodeLineGoal

__all__ = ["NodeSegmentGoal"]


class NodeSegmentGoal(NodeLineGoal):
    """
    Pull a node onto a target segment, defined by its two endpoints.
    """

    def goal(
        self,
        target: Float[Array, "2 3"],
        prediction: Float[Array, "3"],
    ) -> Float[Array, "3"]:
        """
        The point on the target segment closest to the node.

        Parameters
        ----------
        target :
            The two endpoints of the target segment.
        prediction :
            The current node coordinates.

        Returns
        -------
        goal :
            The closest point on the segment, clamped to lie between the endpoints.
        """
        return closest_point_on_segment(prediction, target)
