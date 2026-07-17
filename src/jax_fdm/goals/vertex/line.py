from jax_fdm.goals.node import NodeLineGoal
from jax_fdm.goals.vertex import VertexGoal


class VertexLineGoal(VertexGoal, NodeLineGoal):
    """
    Pull a vertex onto a target line, defined by two points.

    Notes
    -----
    A thin vertex counterpart of :class:`NodeLineGoal`: the goal logic is
    inherited unchanged, while keys resolve against the vertices of a mesh.
    """
