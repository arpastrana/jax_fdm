import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.network import NetworkConstraint
from jax_fdm.equilibrium import EquilibriumState


class NetworkEdgesLengthConstraint(NetworkConstraint):
    """
    Set constraint bounds to the length of all the edges of a network.
    """
    @staticmethod
    def constraint(eqstate: EquilibriumState, model: Int[Array, ""]) -> Float[Array, "edges"]:
        """
        The constraint function relative to a equilibrium state.
        """
        return jnp.ravel(eqstate.lengths)
