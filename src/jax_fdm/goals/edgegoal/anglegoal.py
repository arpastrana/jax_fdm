import numpy as np
import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edgegoal import EdgesGoal


class EdgesVectorAngleGoal(ScalarGoal, EdgesGoal):
    """
    Make the direction of the edge of a network to be parallel to a target vector.
    """
    def __init__(self, keys, vectors, targets, weights=1.0):
        super().__init__(keys=keys, targets=targets, weights=weights)
        self.vectors_other = vectors

    def prediction(self, eq_state):
        """
        The edge vector in the network.
        """
        vectors = eq_state.vectors[self.index, :]
        return self._angle_vectors(vectors, np.reshape(np.array(self.vectors_other), (1, -1)))

    @staticmethod
    def _angle_vectors(u, v):
        """
        Compute the smallest angle between two vectors.
        """
        print("u shape, vshape", u.shape, v.shape)
        L = jnp.linalg.norm(u, axis=-1, keepdims=True) * jnp.linalg.norm(v, axis=-1)
        print("L shape", L.shape)
        a = jnp.dot(u, jnp.transpose(v)) / L
        print("a shape", a.shape)
        a = jnp.maximum(jnp.minimum(a, 1.0), -1.0)
        print("a shape", a.shape)
        return jnp.degrees(jnp.arccos(a))
