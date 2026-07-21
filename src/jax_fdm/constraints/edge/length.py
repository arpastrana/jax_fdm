from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.edge.edge import EdgeConstraint
from jax_fdm.equilibrium import EquilibriumState


class EdgeLengthConstraint(EdgeConstraint):
    """
    Bound the length of an edge between a lower and an upper value.
    """

    def constraint(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The length of the edge.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the length from.
        index :
            The index of the edge.

        Returns
        -------
        constraint :
            The edge's length.
        """
        return eq_state.lengths[index, 0]
