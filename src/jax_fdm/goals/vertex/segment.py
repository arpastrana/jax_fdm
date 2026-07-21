from jax_fdm.goals.node.segment import NodeSegmentGoal
from jax_fdm.goals.vertex.vertex import VertexGoal


class VertexSegmentGoal(VertexGoal, NodeSegmentGoal):
    """
    Pull a vertex onto a target segment, defined by its two endpoints.
    """
