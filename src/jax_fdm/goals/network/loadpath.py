import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.network.network import NetworkGoal

__all__ = ["NetworkLoadPathGoal"]


class NetworkLoadPathGoal(NetworkGoal):
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
        structure: EquilibriumStructure,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The total load path of the network.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read edge forces and lengths from.
        structure :
            The structure the goal is evaluated against; unused.
        index :
            The sentinel index, unused.

        Returns
        -------
        prediction :
            The sum of the absolute force-length product over all edges.
        """
        return jnp.sum(jnp.abs(jnp.multiply(eq_state.lengths, eq_state.forces)))
