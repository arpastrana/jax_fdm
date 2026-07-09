import jax.numpy as jnp
import numpy as np

from jax_fdm.geometry import cosine_vectors
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


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

    def init(self, model, structure):
        """
        Initialize the goal with information from an equilibrium model.
        """
        super().init(model, structure)

        # create matrix of vectors
        vector = self.vector
        vm = np.zeros((max(self.index) + 1, 3))
        for v, idx in zip(vector, self.index):
            vm[idx, :] = v
        self.vector = vm

    def prediction(self, eq_state, index):
        """
        The angle between the edge and the reference vector.
        """
        vector = eq_state.vectors[index, :]

        # clip guards the arccos value and gradient, which are singular when
        # the edge is parallel to the reference vector
        cosine = cosine_vectors(vector, self.vector[index, :])
        angle = jnp.arccos(jnp.clip(cosine, -1.0, 1.0))

        return jnp.atleast_1d(angle)
