import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.network import NetworkConstraint
from jax_fdm.equilibrium import EquilibriumState


class NetworkEdgesLengthConstraint(NetworkConstraint):
    """
    Bound the length of every edge of a network.
    """

    def constraint(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "edges"]:
        """
        The length of every edge.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the lengths from.
        index :
            The sentinel index, unused.

        Returns
        -------
        constraint :
            The length of each edge, flattened.
        """
        return jnp.ravel(eq_state.lengths)
