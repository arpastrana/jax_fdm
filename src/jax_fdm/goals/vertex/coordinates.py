from jax_fdm.goals.node.coordinates import NodeXCoordinateGoal
from jax_fdm.goals.node.coordinates import NodeYCoordinateGoal
from jax_fdm.goals.node.coordinates import NodeZCoordinateGoal
from jax_fdm.goals.vertex.vertex import VertexGoal


class VertexXCoordinateGoal(VertexGoal, NodeXCoordinateGoal):
    """
    Drive a vertex toward a target X coordinate.
    """


class VertexYCoordinateGoal(VertexGoal, NodeYCoordinateGoal):
    """
    Drive a vertex toward a target Y coordinate.
    """


class VertexZCoordinateGoal(VertexGoal, NodeZCoordinateGoal):
    """
    Drive a vertex toward a target Z coordinate.
    """
