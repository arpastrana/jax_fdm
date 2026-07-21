from jax_fdm.goals.node.point import NodePointGoal
from jax_fdm.goals.vertex.vertex import VertexGoal

__all__ = ["VertexPointGoal"]


class VertexPointGoal(VertexGoal, NodePointGoal):
    """
    Drive a vertex toward target xyz coordinates.
    """
