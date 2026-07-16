import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeLengthGoal(ScalarGoal, EdgeGoal):
    """
    Drive an edge toward a target length.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        The current length of the edge.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the length from.
        index :
            The index of the edge.

        Returns
        -------
        prediction :
            The edge's length.
        """
        return eq_state.lengths[index]


class EdgesLengthEqualGoal(ScalarGoal, EdgeGoal):
    """
    Equalize the lengths of a selection of edges.

    Notes
    -----
    Applies to a collection of edges at once, so it is not collectible. The goal
    drives the normalized variance of the lengths to zero.
    """

    def __init__(self, key: list[tuple[int, int]], weight: float = 1.0) -> None:
        super().__init__(key=key, target=0.0, weight=weight)
        self.is_collectible = False

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Bind the goal to a structure, keeping the edge indices as one collection.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure whose edge ordering defines the indices.
        """
        self.index = np.atleast_2d(super().index_from_model(model, structure))

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, "elements"],
    ) -> Float[Array, "1"]:
        """
        The variance of the edge lengths, normalized by their mean.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the lengths from.
        index :
            The indices of the edges.

        Returns
        -------
        prediction :
            The mean-normalized variance of the lengths, zero when all equal.
        """
        lengths = eq_state.lengths[index]

        return jnp.atleast_1d(jnp.var(lengths) / jnp.mean(lengths))
