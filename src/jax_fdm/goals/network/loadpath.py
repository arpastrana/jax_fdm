import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.network import NetworkGoal


class NetworkLoadPathGoal(ScalarGoal, NetworkGoal):
    """
    Make the total load path of a network to reach a target magnitude.

    The load path of an edge is the absolute value of the product of the
    the force on the edge time its length.
    """
    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
        ) -> Float[Array, "1"]:
        """
        The current load path of the network.
        """
        load_path = jnp.sum(jnp.abs(jnp.multiply(eq_state.lengths, eq_state.forces)))

        return jnp.atleast_1d(load_path)
