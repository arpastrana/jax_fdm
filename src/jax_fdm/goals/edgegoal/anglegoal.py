import numpy as np
import jax.numpy as jnp

from jax_fdm.geometry import angle_vectors

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edgegoal import EdgeGoal


class EdgeAngleGoal(ScalarGoal, EdgeGoal):
    """
    Reach a target angle between the direction of an edge and a reference vector.
    """
    def __init__(self, key, vector, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)
        self._vector = None
        self.vector = vector

    @property
    def vector(self):
        """
        The vector to take the angle with.
        """
        return self._vector

    @vector.setter
    def vector(self, vector):
        self._vector = jnp.reshape(jnp.asarray(vector), (-1, 3))

    def prediction(self, eq_state, index):
        """
        The angle between the edge and the reference vector.
        """
        vector = eq_state.vectors[index, :]
        return angle_vectors(vector, self.vector[index, :])


if __name__ == "__main__":

    vector = [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
    # vector = np.array([0.0, 0.0, 1.0])
    us = np.reshape(np.array(vector), (-1, 3))

    vectors_other = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]
    vs = np.reshape(np.array(vectors_other), (-1, 3))

    # target = 30.0
    # keys = [(0, 1), (1, 2)]
    # goal = EdgeAngleGoal(keys, vector, target)

    for u, v in zip(us, vs):
        # breakpoint()
        angle = angle_vectors(u, v)
        print(angle.shape)
        print(angle)


    # @staticmethod
    # def _angle_vectors(u, v):
    #     """
    #     Compute the smallest angle between two vectors.
    #     """
    #     L = jnp.linalg.norm(u) * jnp.linalg.norm(v)
    #     cosim = (u @ v) / L
    #     cosim = jnp.maximum(jnp.minimum(cosim, -1.0), -1.0)

    #     return jnp.degrees(jnp.arccos(cosim))

    # @staticmethod
    # def prediction(eq_state, index, gattrs):
    #     """
    #     The edge vector in the network.
    #     """
    #     vector = eq_state.vectors[index, :]
    #     return angle_vectors(vector, gattrs["vector"])

    # def _angle_vectors(u, v):
    #     """
    #     Compute the smallest angle between two vectors.
    #     """
    #     L = jnp.linalg.norm(u, axis=-1) * jnp.linalg.norm(v, axis=-1)
    #     a = jnp.einsum('ij,ij->i', u, v) / L
    #     a = jnp.maximum(jnp.minimum(a, 1.0), -1.0)

    #     return jnp.degrees(jnp.arccos(a))
