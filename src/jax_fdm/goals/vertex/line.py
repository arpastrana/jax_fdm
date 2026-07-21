from jax_fdm.goals.node.line import NodeLineGoal
from jax_fdm.goals.vertex.vertex import VertexGoal


class VertexLineGoal(VertexGoal, NodeLineGoal):
    """
    Pull a vertex onto a target line, defined by two points.
    """
