from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.edge.edge import EdgeGoal

__all__ = [
    "EdgeForceGoal",
    "EdgesForceEqualGoal",
]


class EdgeForceGoal(EdgeGoal):
    """
    Drive an edge toward a target internal force.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumStructure,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The current internal force in the edge.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the force from.
        structure :
            The structure the goal is evaluated against; unused.
        index :
            The index of the edge.

        Returns
        -------
        prediction :
            The edge's internal force, signed positive in tension.
        """
        return eq_state.forces[index, 0]


class EdgesForceEqualGoal(EdgeGoal):
    """
    Equalize the internal forces of a selection of edges.

    Notes
    -----
    Applies to a collection of edges at once. The goal drives the normalized
    variance of the forces to zero.
    """

    is_aggregate = True

    def __init__(self, key: Sequence[tuple[int, int]], weight: float = 1.0) -> None:
        super().__init__(key=key, target=0.0, weight=weight)

    def prediction(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumStructure,
        index: Int[Array, "elements"],
    ) -> Float[Array, ""]:
        """
        The variance of the edge forces, normalized by their mean magnitude.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the forces from.
        structure :
            The structure the goal is evaluated against; unused.
        index :
            The indices of the edges.

        Returns
        -------
        prediction :
            The magnitude-normalized variance of the forces, zero when all equal.
        """
        forces = eq_state.forces[index]

        return jnp.var(forces) / jnp.mean(jnp.abs(forces))
