import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.network.network import NetworkConstraint
from jax_fdm.equilibrium import EquilibriumState


class NetworkEdgesForceConstraint(NetworkConstraint):
    """
    Bound the internal force of every edge of a network.
    """

    def constraint(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "edges"]:
        """
        The internal force in every edge.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the forces from.
        index :
            The sentinel index, unused.

        Returns
        -------
        constraint :
            The internal force of each edge, flattened.
        """
        return jnp.ravel(eq_state.forces)
