import numpy as np

import jax.numpy as jnp

from jax_fdm.geometry import angle_vectors

from jax_fdm.constraints.edge import EdgeConstraint


class EdgeAngleConstraint(EdgeConstraint):
    """
    Constraints the angle formed by an edge and a vector between a lower and an upper bound.
    """
    def __init__(self, key, vector, bound_low, bound_up):
        super().__init__(key=key, bound_low=bound_low, bound_up=bound_up)
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

    def vectors(self):
        """
        Create a matrix of vectors.
        """
        matrix = np.zeros((max(self.index) + 1, 3))
        for vec, idx in zip(self.vector, self.index):
            matrix[idx, :] = vec
        return matrix

    def init(self, model, structure):
        """
        Initialize the constraint with information from an equilibrium model.
        """
        super().init(model, structure)
        self.vector = self.vectors()

    def constraint(self, eqstate, index):
        """
        Returns the angle between an edge in an equilibrium state and a vector.
        """
        vector = eqstate.vectors[index, :]
        return angle_vectors(vector, self.vector[index, :])
