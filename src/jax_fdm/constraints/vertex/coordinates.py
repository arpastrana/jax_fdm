from jax_fdm.constraints.node import NodeXCoordinateConstraint
from jax_fdm.constraints.node import NodeYCoordinateConstraint
from jax_fdm.constraints.node import NodeZCoordinateConstraint
from jax_fdm.constraints.vertex import VertexConstraint


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
