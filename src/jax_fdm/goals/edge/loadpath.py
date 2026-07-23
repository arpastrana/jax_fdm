import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.edge.edge import EdgeGoal

__all__ = ["EdgeLoadPathGoal"]


class EdgeLoadPathGoal(EdgeGoal):
    """
    Drive an edge's load path toward a target value.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumStructure,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The load path of the edge.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the force and length from.
        structure :
            The structure the goal is evaluated against; unused.
        index :
            The index of the edge.

        Returns
        -------
        prediction :
            The edge load path, the product of absolute force and length.
        """
        return jnp.abs(eq_state.forces[index, 0]) * eq_state.lengths[index, 0]
