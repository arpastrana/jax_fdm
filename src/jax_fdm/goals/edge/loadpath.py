import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeLoadPathGoal(ScalarGoal, EdgeGoal):
    """
    Make an edge of a network to reach a target force.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        The predicted edge force.
        """
        return jnp.abs(eq_state.forces[index]) * eq_state.lengths[index]
