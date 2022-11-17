from jax_fdm.geometry import normalize_vector

from jax_fdm.goals import VectorGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeDirectionGoal(VectorGoal, EdgeGoal):
    """
    Make the direction of the edge of a network to be parallel to a target vector.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The edge vector in the network.
        """
        vector = eq_state.vectors[index, :]
        return normalize_vector(vector)

    @staticmethod
    def goal(target, prediction):
        """
        The target vector.
        """
        return normalize_vector(target)
