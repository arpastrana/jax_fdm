from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.geometry import closest_point_on_segment
from jax_fdm.goals.node import NodeLineGoal


class NodeSegmentGoal(NodeLineGoal):
    """
    Pulls the xyz position of a node to a target line ray.
    """

    def goal(
        self,
        target: Float[Array, "2 3"],
        prediction: Float[Array, "3"],
    ) -> Float[Array, "3"]:
        """
        The closes point on the target seegment.
        """
        return closest_point_on_segment(prediction, target)
