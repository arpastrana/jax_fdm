from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.edge import EdgeConstraint
from jax_fdm.equilibrium import EquilibriumState


class EdgeForceConstraint(EdgeConstraint):
    """
    Constraints the force of an edge between a lower and an upper bound.
    """
    def constraint(
        self,
        eqstate: EquilibriumState,
        index: Int[Array, ""],
        ) -> Float[Array, ""]:
        """
        Returns the force of an edge from an equilibrium state.
        """
        return eqstate.forces[index, 0]
