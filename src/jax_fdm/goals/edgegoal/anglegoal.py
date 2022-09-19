import numpy as np
import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edgegoal import EdgesGoal

from jax import jit


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
        others = np.reshape(np.array(self.vectors_other), (-1, 3))
        return self._angle_vectors(vectors, others)

    @staticmethod
    @jit
    def _angle_vectors(u, v):
        """
        Compute the smallest angle between two vectors.
        """
        L = jnp.linalg.norm(u, axis=-1) * jnp.linalg.norm(v, axis=-1)
        a = jnp.einsum('ij,ij->i', u, v) / L
        a = jnp.maximum(jnp.minimum(a, 1.0), -1.0)

        return jnp.degrees(jnp.arccos(a))


if __name__ == "__main__":

    vector = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
    v = np.array(vector)

    vector = np.array([0.0, 0.0, 1.0])
    # v = vector
    v = np.reshape(np.array(vector), (-1, 3))

    target = 30.0
    weight = 1

    keys = [(0, 1), (1, 2)]

    goal = EdgesVectorAngleGoal(keys, vector, target, weight)

    u = jnp.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])

    angles = goal._angle_vectors(u, v)
    # angles = goal._angle_vectors_vmap(u, v)
    print(angles.shape)

    print(angles)
