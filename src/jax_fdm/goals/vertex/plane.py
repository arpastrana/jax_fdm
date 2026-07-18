from jax_fdm.goals.node import NodePlaneGoal
from jax_fdm.goals.vertex import VertexGoal


class VertexPlaneGoal(VertexGoal, NodePlaneGoal):
    """
    Pull a vertex onto a target plane, defined by a point and a normal.
    """
