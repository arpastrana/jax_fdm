from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.edge.edge import EdgeGoal

__all__ = [
    "EdgeLengthGoal",
    "EdgesLengthEqualGoal",
]


class EdgeLengthGoal(EdgeGoal):
    """
    Drive an edge toward a target length.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumStructure,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The current length of the edge.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the length from.
        structure :
            The structure the goal is evaluated against; unused.
        index :
            The index of the edge.

        Returns
        -------
        prediction :
            The edge's length.
        """
        return eq_state.lengths[index, 0]


class EdgesLengthEqualGoal(EdgeGoal):
    """
    Equalize the lengths of a selection of edges.

    Notes
    -----
    Applies to a collection of edges at once. The goal drives the normalized
    variance of the lengths to zero.
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
        The variance of the edge lengths, normalized by their mean.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the lengths from.
        structure :
            The structure the goal is evaluated against; unused.
        index :
            The indices of the edges.

        Returns
        -------
        prediction :
            The mean-normalized variance of the lengths, zero when all equal.
        """
        lengths = eq_state.lengths[index]

        return jnp.var(lengths) / jnp.mean(lengths)
