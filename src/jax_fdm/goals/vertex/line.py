from jax_fdm.goals.node import NodeLineGoal
from jax_fdm.goals.vertex import VertexGoal


class VertexLineGoal(VertexGoal, NodeLineGoal):
    """
    Pull a vertex onto a target line, defined by two points.
    """
