from jax_fdm.constraints.node.coordinates import NodeXCoordinateConstraint
from jax_fdm.constraints.node.coordinates import NodeYCoordinateConstraint
from jax_fdm.constraints.node.coordinates import NodeZCoordinateConstraint
from jax_fdm.constraints.vertex.vertex import VertexConstraint

__all__ = [
    "VertexXCoordinateConstraint",
    "VertexYCoordinateConstraint",
    "VertexZCoordinateConstraint",
]


class VertexXCoordinateConstraint(VertexConstraint, NodeXCoordinateConstraint):
    """
    Bound the X coordinate of a vertex between a lower and an upper value.
    """


class VertexYCoordinateConstraint(VertexConstraint, NodeYCoordinateConstraint):
    """
    Bound the Y coordinate of a vertex between a lower and an upper value.
    """


class VertexZCoordinateConstraint(VertexConstraint, NodeZCoordinateConstraint):
    """
    Bound the Z coordinate of a vertex between a lower and an upper value.
    """
