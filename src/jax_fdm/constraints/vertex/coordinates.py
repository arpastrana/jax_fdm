from jax_fdm.constraints.node import NodeXCoordinateConstraint
from jax_fdm.constraints.node import NodeYCoordinateConstraint
from jax_fdm.constraints.node import NodeZCoordinateConstraint
from jax_fdm.constraints.vertex import VertexConstraint


class VertexXCoordinateConstraint(VertexConstraint, NodeXCoordinateConstraint):
    """
    Bound the X coordinate of a vertex between a lower and an upper value.

    Notes
    -----
    A thin vertex counterpart of :class:`NodeXCoordinateConstraint`: the
    constraint logic is inherited unchanged, while keys resolve against the
    vertices of a mesh.
    """


class VertexYCoordinateConstraint(VertexConstraint, NodeYCoordinateConstraint):
    """
    Bound the Y coordinate of a vertex between a lower and an upper value.

    Notes
    -----
    A thin vertex counterpart of :class:`NodeYCoordinateConstraint`: the
    constraint logic is inherited unchanged, while keys resolve against the
    vertices of a mesh.
    """


class VertexZCoordinateConstraint(VertexConstraint, NodeZCoordinateConstraint):
    """
    Bound the Z coordinate of a vertex between a lower and an upper value.

    Notes
    -----
    A thin vertex counterpart of :class:`NodeZCoordinateConstraint`: the
    constraint logic is inherited unchanged, while keys resolve against the
    vertices of a mesh.
    """
