"""Goals defined on the residual forces at mesh vertices."""

from jax_fdm.goals.node import NodeResidualDirectionGoal
from jax_fdm.goals.node import NodeResidualForceGoal
from jax_fdm.goals.node import NodeResidualVectorGoal
from jax_fdm.goals.vertex import VertexGoal


class VertexResidualForceGoal(VertexGoal, NodeResidualForceGoal):
    """
    Drive the residual force magnitude at a vertex toward a target value.

    Notes
    -----
    A thin vertex counterpart of :class:`NodeResidualForceGoal`: the goal logic
    is inherited unchanged, while keys resolve against the vertices of a mesh.
    """


class VertexResidualVectorGoal(VertexGoal, NodeResidualVectorGoal):
    """
    Drive the residual force at a vertex toward a target vector.

    Notes
    -----
    A thin vertex counterpart of :class:`NodeResidualVectorGoal`: the goal logic
    is inherited unchanged, while keys resolve against the vertices of a mesh.
    """


class VertexResidualDirectionGoal(VertexGoal, NodeResidualDirectionGoal):
    """
    Drive the residual force at a vertex toward a target direction.

    Notes
    -----
    A thin vertex counterpart of :class:`NodeResidualDirectionGoal`: the goal
    logic is inherited unchanged, while keys resolve against the vertices of a
    mesh. Both the prediction and the target are unit-normalized, so only
    direction is compared while magnitude is ignored.
    """
