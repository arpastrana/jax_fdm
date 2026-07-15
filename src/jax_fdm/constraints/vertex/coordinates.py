from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.vertex import VertexConstraint
from jax_fdm.equilibrium import EquilibriumState


class VertexXCoordinateConstraint(VertexConstraint):
    """
    Constraint the X coordinate of a vertex between a lower and an upper bound.
    """
    def constraint(self, eqstate: EquilibriumState, index: Int[Array, ""]) -> Float[Array, ""]:
        """
        Returns the X coordinate of a vertex from an equilibrium state.
        """
        return eqstate.xyz[index, 0]


class VertexYCoordinateConstraint(VertexConstraint):
    """
    Constraint the Y coordinate of a vertex between a lower and an upper bound.
    """
    def constraint(self, eqstate: EquilibriumState, index: Int[Array, ""]) -> Float[Array, ""]:
        """
        Returns the Y coordinate of a vertex from an equilibrium state.
        """
        return eqstate.xyz[index, 1]


class VertexZCoordinateConstraint(VertexConstraint):
    """
    Constraint the Z coordinate of a vertex between a lower and an upper bound.
    """
    def constraint(self, eqstate: EquilibriumState, index: Int[Array, ""]) -> Float[Array, ""]:
        """
        Returns the Z coordinate of a vertex from an equilibrium state.
        """
        return eqstate.xyz[index, 2]
