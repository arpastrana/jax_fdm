from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.geometry import closest_point_on_plane
from jax_fdm.goals.node.point import NodePointGoal

__all__ = ["NodePlaneGoal"]


class NodePlaneGoal(NodePointGoal):
    """
    Pull a node onto a target plane, defined by a point and a normal.

    Notes
    -----
    The target is the plane's origin and normal, one ``(2, 3)`` array per
    element; `tree_stack` adds the leading element axis when goals are grouped.
    """

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
