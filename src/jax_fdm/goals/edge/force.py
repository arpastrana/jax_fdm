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


class EdgeForceGoal(ScalarGoal, EdgeGoal):
    """
    Make an edge of a network to reach a target force.
    """
    def prediction(self, eq_state: EquilibriumState, index: Int[Array, ""]) -> Float[Array, "1"]:
        """
        The predicted edge force.
        """
        return eq_state.forces[index]


class EdgesForceEqualGoal(ScalarGoal, EdgeGoal):
    """
    Equalize the internal force in a selection of edges by minimizing
    the normalized variance of their internal forces.
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
        The normalized variance of the forces of the edges.
        """
        forces = eq_state.forces[index]

        return jnp.atleast_1d(jnp.var(forces) / jnp.mean(jnp.abs(forces)))
