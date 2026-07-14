from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.node import NodeConstraint
from jax_fdm.equilibrium import EquilibriumState


class NodeXCoordinateConstraint(NodeConstraint):
    """
    Constraint the X coordinate of a node between a lower and an upper bound.
    """
    @staticmethod
    def constraint(eqstate: EquilibriumState, index: Int[Array, ""]) -> Float[Array, "1"]:
        """
        Returns the X coordinate of a node from an equilibrium state.
        """
        return eqstate.xyz[index, :1]


class NodeYCoordinateConstraint(NodeConstraint):
    """
    Constraint the Y coordinate of a node between a lower and an upper bound.
    """
    @staticmethod
    def constraint(eqstate: EquilibriumState, index: Int[Array, ""]) -> Float[Array, "1"]:
        """
        Returns the Y coordinate of a node from an equilibrium state.
        """
        return eqstate.xyz[index, 1:2]


class NodeZCoordinateConstraint(NodeConstraint):
    """
    Constraint the Z coordinate of a node between a lower and an upper bound.
    """
    @staticmethod
    def constraint(eqstate: EquilibriumState, index: Int[Array, ""]) -> Float[Array, ""]:
        """
        Returns the Z coordinate of a node from an equilibrium state.
        """
        return eqstate.xyz[index, 2]
