from jax_fdm.goals.node.plane import NodePlaneGoal
from jax_fdm.goals.vertex.vertex import VertexGoal

__all__ = ["VertexPlaneGoal"]


class VertexPlaneGoal(VertexGoal, NodePlaneGoal):
    """
    Pull a vertex onto a target plane, defined by a point and a normal.
    """
