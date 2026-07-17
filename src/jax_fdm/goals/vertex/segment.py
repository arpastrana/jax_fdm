from jax_fdm.goals.node import NodeSegmentGoal
from jax_fdm.goals.vertex import VertexGoal


class VertexSegmentGoal(VertexGoal, NodeSegmentGoal):
    """
    Pull a vertex onto a target segment, defined by its two endpoints.

    Notes
    -----
    A thin vertex counterpart of :class:`NodeSegmentGoal`: the goal logic is
    inherited unchanged, while keys resolve against the vertices of a mesh.
    """
