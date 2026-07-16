import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeLoadPathGoal(ScalarGoal, EdgeGoal):
    """
    Drive an edge's load path toward a target value.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        The load path of the edge.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the force and length from.
        index :
            The index of the edge.

        Returns
        -------
        prediction :
            The edge load path, the product of absolute force and length.
        """
        return jnp.abs(eq_state.forces[index]) * eq_state.lengths[index]
