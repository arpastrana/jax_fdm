from jax_fdm.goals.node import NodeXCoordinateGoal
from jax_fdm.goals.node import NodeYCoordinateGoal
from jax_fdm.goals.node import NodeZCoordinateGoal
from jax_fdm.goals.vertex import VertexGoal


class VertexXCoordinateGoal(VertexGoal, NodeXCoordinateGoal):
    """
    Drive a vertex toward a target X coordinate.

    Notes
    -----
    A thin vertex counterpart of :class:`NodeXCoordinateGoal`: the goal logic is
    inherited unchanged, while keys resolve against the vertices of a mesh.
    """


class VertexYCoordinateGoal(VertexGoal, NodeYCoordinateGoal):
    """
    Drive a vertex toward a target Y coordinate.

    Notes
    -----
    A thin vertex counterpart of :class:`NodeYCoordinateGoal`: the goal logic is
    inherited unchanged, while keys resolve against the vertices of a mesh.
    """


class VertexZCoordinateGoal(VertexGoal, NodeZCoordinateGoal):
    """
    Drive a vertex toward a target Z coordinate.

    Notes
    -----
    A thin vertex counterpart of :class:`NodeZCoordinateGoal`: the goal logic is
    inherited unchanged, while keys resolve against the vertices of a mesh.
    """
