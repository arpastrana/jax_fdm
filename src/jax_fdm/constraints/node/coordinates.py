from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.node import NodeConstraint
from jax_fdm.equilibrium import EquilibriumState


class NodeXCoordinateConstraint(NodeConstraint):
    """
    Constraint the X coordinate of a node between a lower and an upper bound.
    """
    def constraint(self, eqstate: EquilibriumState, index: Int[Array, ""]) -> Float[Array, ""]:
        """
        Returns the X coordinate of a node from an equilibrium state.
        """
        return eqstate.xyz[index, 0]


class NodeYCoordinateConstraint(NodeConstraint):
    """
    Constraint the Y coordinate of a node between a lower and an upper bound.
    """
    def constraint(self, eqstate: EquilibriumState, index: Int[Array, ""]) -> Float[Array, ""]:
        """
        Returns the Y coordinate of a node from an equilibrium state.
        """
        return eqstate.xyz[index, 1]


class NodeZCoordinateConstraint(NodeConstraint):
    """
    Constraint the Z coordinate of a node between a lower and an upper bound.
    """
    def constraint(self, eqstate: EquilibriumState, index: Int[Array, ""]) -> Float[Array, ""]:
        """
        Returns the Z coordinate of a node from an equilibrium state.
        """
        return eqstate.xyz[index, 2]
