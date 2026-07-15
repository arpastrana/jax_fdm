import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.network import NetworkConstraint
from jax_fdm.equilibrium import EquilibriumState


class NetworkEdgesForceConstraint(NetworkConstraint):
    """
    Set constraint bounds to the force passing through every edge of a network.
    """
    def constraint(
        self,
        eqstate: EquilibriumState,
        index: Int[Array, ""],
        ) -> Float[Array, "edges"]:
        """
        The constraint function relative to an equilibrium state.
        """
        return jnp.ravel(eqstate.forces)
