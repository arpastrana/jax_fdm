import numpy as np
import jax.numpy as jnp

from jax_fdm.goals import VectorGoal
from jax_fdm.goals.edgegoal import EdgesGoal


class EdgesDirectionGoal(VectorGoal, EdgesGoal):
    """
    Make the direction of the edge of a network to be parallel to a target vector.
    """
    def __init__(self, keys, targets, weights=1.0):
        super().__init__(keys=keys, targets=targets, weights=weights)

    def prediction(self, eq_state):
        """
        The edge vector in the network.
        """
        vector = eq_state.vectors[self.index, :]
        return vector / jnp.linalg.norm(vector, axis=1, keepdims=True)

    def target(self, prediction):
        """
        The target vector.
        """
        return self._target / np.linalg.norm(self._target, axis=-1, keepdims=True)
