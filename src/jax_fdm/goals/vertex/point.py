from jax_fdm.goals.node import NodePointGoal
from jax_fdm.goals.vertex import VertexGoal


class VertexPointGoal(VertexGoal, NodePointGoal):
    """
    Drive a vertex toward target xyz coordinates.

    Notes
    -----
    A thin vertex counterpart of :class:`NodePointGoal`: the goal logic is
    inherited unchanged, while keys resolve against the vertices of a mesh.
    """
