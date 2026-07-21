"""Goals defined on the residual forces at mesh vertices."""

from jax_fdm.goals.node.residual import NodeResidualDirectionGoal
from jax_fdm.goals.node.residual import NodeResidualForceGoal
from jax_fdm.goals.node.residual import NodeResidualPlaneGoal
from jax_fdm.goals.node.residual import NodeResidualVectorGoal
from jax_fdm.goals.vertex.vertex import VertexGoal

__all__ = [
    "VertexResidualDirectionGoal",
    "VertexResidualForceGoal",
    "VertexResidualPlaneGoal",
    "VertexResidualVectorGoal",
]


class VertexResidualForceGoal(VertexGoal, NodeResidualForceGoal):
    """
    Drive the residual force magnitude at a vertex toward a target value.
    """


class VertexResidualVectorGoal(VertexGoal, NodeResidualVectorGoal):
    """
    Drive the residual force at a vertex toward a target vector.
    """


class VertexResidualDirectionGoal(VertexGoal, NodeResidualDirectionGoal):
    """
    Drive the residual force at a vertex toward a target direction.

    Notes
    -----
    Both the prediction and the target are unit-normalized, so only direction
    is compared while magnitude is ignored.
    """


class VertexResidualPlaneGoal(VertexGoal, NodeResidualPlaneGoal):
    """
    Drive the residual vector at a vertex to lie in a target plane.

    Notes
    -----
    The plane passes through the origin and is described by its normal vector
    alone. Only the residual's direction is compared, so the error cannot be
    reduced by shrinking the reaction magnitude instead of rotating the
    reaction into the plane.
    """
