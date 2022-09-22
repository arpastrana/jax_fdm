import numpy as np
import jax.numpy as jnp

from jax_fdm.goals import VectorGoal
from jax_fdm.goals.edgegoal import EdgeGoal


class EdgeDirectionGoal(VectorGoal, EdgeGoal):
    """
    Make the direction of the edge of a network to be parallel to a target vector.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    @staticmethod
    def prediction(eq_state, index):
        """
        The edge vector in the network.
        """
        vector = eq_state.vectors[index, :]
        return vector / jnp.linalg.norm(vector, axis=-1)

    @staticmethod
    def goal(target, prediction):
        """
        The target vector.
        """
        return target / jnp.linalg.norm(target, axis=-1)
