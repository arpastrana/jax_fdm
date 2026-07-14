from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.geometry import normalize_vector
from jax_fdm.goals import VectorGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeDirectionGoal(VectorGoal, EdgeGoal):
    """
    Make the direction of the edge of a network to be parallel to a target vector.
    """
    @staticmethod
    def prediction(eq_state: EquilibriumState, index: Int[Array, ""]) -> Float[Array, "3"]:
        """
        The edge vector in the network.
        """
        vector = eq_state.vectors[index, :]
        return normalize_vector(vector)

    @staticmethod
    def goal(target: Float[Array, "3"], prediction: Float[Array, "3"]) -> Float[Array, "3"]:
        """
        The target vector.
        """
        return normalize_vector(target)
