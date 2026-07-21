from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.edge.edge import EdgeConstraint
from jax_fdm.equilibrium import EquilibriumState

__all__ = ["EdgeForceConstraint"]


class EdgeForceConstraint(EdgeConstraint):
    """
    Bound the internal force of an edge between a lower and an upper value.
    """

    def constraint(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The internal force in the edge.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the force from.
        index :
            The index of the edge.

        Returns
        -------
        constraint :
            The edge's internal force, signed positive in tension.
        """
        return eq_state.forces[index, 0]
