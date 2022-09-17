import numpy as np
import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edgegoal import EdgeGoal


class EdgeVectorAngleGoal(ScalarGoal, EdgeGoal):
    """
    Make the direction of the edge of a network to be parallel to a target vector.
    """
    def __init__(self, key, vector, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)
        self.vector_other = np.asarray(vector)

    def prediction(self, eq_state):
        """
        The edge vector in the network.
        """
        vector = eq_state.vectors[self.index, :]
        angle = self._angle_vectors_numpy(vector, self.vector_other, deg=True)
        return np.atleast_1d(angle)

    @staticmethod
    def _angle_vectors_numpy(u, v, deg=False):
        """
        Compute the smallest angle between two vectors.
        """
        a = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
        a = max(min(a, 1), -1)
        if deg:
            return np.degrees(np.arccos(a))
        return np.arccos(a)
