import numpy as np
import jax.numpy as jnp

from jax_fdm.constraints import Constraint


class EdgeConstraint(Constraint):
    """
    Base class for all constraints that pertain to an edge of a network.
    """
    def __init__(self, key, bound_low, bound_up, **kwargs):
        super().__init__(bound_low=bound_low, bound_up=bound_up)
        self._key = key

    def index(self, model):
        """
        The index of the edge key in a structure.
        """
        return model.structure.edge_index[self.key()]

    def key(self):
        """
        The key of the edge in the network.
        """
        return self._key


class EdgeLengthConstraint(EdgeConstraint):
    """
    Constraints the length of an edge between a lower and an upper bound.
    """
    def constraint(self, eqstate, model):
        """
        Returns the length of an edge from an equilibrium state.
        """
        return eqstate.lengths[self.index(model)]


class EdgeVectorAngleConstraint(EdgeConstraint):
    """
    Constraints the angle formed by an edge and a vector between a lower and an upper bound.
    """
    def __init__(self, key, vector, bound_low, bound_up):
        super().__init__(key=key, bound_low=bound_low, bound_up=bound_up)
        self.vector_other = np.asarray(vector)

    def constraint(self, eqstate, model):
        """
        Returns the angle between an edge in an equilibrium state and a vector.
        """
        vector = eqstate.vectors[self.index(model)]
        return self._angle_vectors_numpy(vector, self.vector_other, deg=True)

    @staticmethod
    def _angle_vectors_numpy(u, v, deg=False):
        """
        Compute the smallest angle between two vectors.
        """
        a = jnp.dot(u, v) / (jnp.linalg.norm(u) * jnp.linalg.norm(v))
        a = max(min(a, 1), -1)
        if deg:
            return jnp.degrees(jnp.arccos(a))
        return jnp.arccos(a)
