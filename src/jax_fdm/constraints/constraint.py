import numpy as np

from jax import vmap

import jax.numpy as jnp


class Constraint:
    """
    Base class for all constraints.
    """
    def __init__(self, key, bound_low=None, bound_up=None):
        self._key = None
        self.key = key

        self._bound_low = None
        self.bound_low = bound_low

        self._bound_up = None
        self.bound_up = bound_up

        self._index = None

    @property
    def key(self):
        """
        The key of an element in a network.
        """
        return self._key

    @key.setter
    def key(self, key):
        self._key = key

    @property
    def index(self):
        """
        The index of the goal key in the canonical ordering of an equilibrium structure.
        """
        return self._index

    @index.setter
    def index(self, index):
        if isinstance(index, int):
            index = [index]
        self._index = np.array(index)

    @staticmethod
    def _bound_setter(bound):
        """
        Set a bound.
        """
        if not isinstance(bound, (int, float)):
            if len(bound) == 1:
                bound = bound[0]
            else:
                bound = np.ravel(bound)
        return bound

    @property
    def bound_low(self):
        """
        The lower bound of this constraint.
        """
        return self._bound_low

    @bound_low.setter
    def bound_low(self, bound):
        if bound is None:
            bound = -jnp.inf
        self._bound_low = self._bound_setter(bound)

    @property
    def bound_up(self):
        """
        The upper bound of this constraint.
        """
        return self._bound_up

    @bound_up.setter
    def bound_up(self, bound):
        if bound is None:
            bound = jnp.inf
        self._bound_up = self._bound_setter(bound)

    @staticmethod
    def index_from_model(model):
        """
        Get the index in the model of the constraint key.
        """
        raise NotImplementedError

    def init(self, model):
        """
        Initialize the constraint with information from an equilibrium model.
        """
        self.index = self.index_from_model(model)

    def __call__(self, q, xyz_fixed, loads, model):
        """
        The called constraint function.
        """
        eqstate = model(q, xyz_fixed, loads)
        constraint = vmap(self.constraint, in_axes=(None, 0))(eqstate, self.index)

        # assert jnp.ravel(constraint).shape == jnp.ravel(self.index).shape, f"Constraint shape: {constraint.shape} vs. index shape: {self.index.shape}"

        return jnp.ravel(constraint)

    def constraint(self, eqstate, index):
        """
        The constraint function.
        """
        raise NotImplementedError
