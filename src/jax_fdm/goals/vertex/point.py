from jax_fdm.goals.node import NodePointGoal
from jax_fdm.goals.vertex import VertexGoal


class VertexPointGoal(VertexGoal, NodePointGoal):
    """
    Drive a vertex toward target xyz coordinates.
    """
