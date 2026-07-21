import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals.edge.edge import EdgeGoal
from jax_fdm.goals.goal import ScalarGoal

__all__ = [
    "EdgeForceGoal",
    "EdgesForceEqualGoal",
]


class EdgeForceGoal(ScalarGoal, EdgeGoal):
    """
    Drive an edge toward a target internal force.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        The current internal force in the edge.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the force from.
        index :
            The index of the edge.

        Returns
        -------
        prediction :
            The edge's internal force, signed positive in tension.
        """
        return eq_state.forces[index]


class EdgesForceEqualGoal(ScalarGoal, EdgeGoal):
    """
    Equalize the internal forces of a selection of edges.

    Notes
    -----
    Applies to a collection of edges at once. The goal drives the normalized
    variance of the forces to zero.
    """

    is_aggregate = True

    def __init__(self, key: list[tuple[int, int]], weight: float = 1.0) -> None:
        super().__init__(key=key, target=0.0, weight=weight)

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, "elements"],
    ) -> Float[Array, ""]:
        """
        The variance of the edge forces, normalized by their mean magnitude.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the forces from.
        index :
            The indices of the edges.

        Returns
        -------
        prediction :
            The magnitude-normalized variance of the forces, zero when all equal.
        """
        forces = eq_state.forces[index]

        return jnp.var(forces) / jnp.mean(jnp.abs(forces))
