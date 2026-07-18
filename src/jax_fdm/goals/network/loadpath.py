import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.network import NetworkGoal


class NetworkLoadPathGoal(ScalarGoal, NetworkGoal):
    """
    Drive the total load path of a network toward a target magnitude.

    Notes
    -----
    The load path of an edge is the absolute value of its force times its length;
    the network load path is the sum over all edges.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, "1"],
    ) -> Float[Array, ""]:
        """
        The total load path of the network.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read edge forces and lengths from.
        index :
            The sentinel index, unused.

        Returns
        -------
        prediction :
            The sum of the absolute force-length product over all edges.
        """
        return jnp.sum(jnp.abs(jnp.multiply(eq_state.lengths, eq_state.forces)))
