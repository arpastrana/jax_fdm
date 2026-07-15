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
    Make an edge of a network to reach a certain length.
    """
    def prediction(self, eq_state: EquilibriumState, index: Int[Array, ""]) -> Float[Array, "1"]:
        """
        The current edge length.
        """
        return eq_state.lengths[index]


class EdgesLengthEqualGoal(ScalarGoal, EdgeGoal):
    """
    Equalize the length of a selection of edges by minimizing
    the normalized variance of their lengths.
    """
    def __init__(self, key: list[tuple[int, int]], weight: float = 1.0):
        super().__init__(key=key, target=0.0, weight=weight)
        self.is_collectible = False

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Initialize the goal with information from an equilibrium model.
        """
        self.index = np.atleast_2d(super().index_from_model(model, structure))

    def prediction(self, eq_state: EquilibriumState, index: Int[Array, "elements"]) -> Float[Array, "1"]:
        """
        The normalized variance of the lengths of the edges.
        """
        lengths = eq_state.lengths[index]

        return jnp.atleast_1d(jnp.var(lengths) / jnp.mean(lengths))
