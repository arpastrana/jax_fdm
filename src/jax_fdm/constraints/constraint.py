from functools import partial

import numpy as np

import jax.numpy as jnp

from jax import jit

from jax import vmap


class Constraint:
    def __init__(self, key, bound_low, bound_up):
        self._key = None
        self._bound_low = None
        self._bound_up = None
        self._index = None

        self.key = key
        self.bound_low = bound_low
        self.bound_up = bound_up

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

    @property
    def bound_low(self):
        """
        The lower bound of this constraint.
        """
        return self._bound_low

    @bound_low.setter
    def bound_low(self, bound):
        if isinstance(bound, (int, float)):
            bound = bound
        elif len(bound) == 1:
            bound = bound[0]
        else:
            bound = np.ravel(bound)
        self._bound_low = bound

    @property
    def bound_up(self):
        """
        The upper bound of this constraint.
        """
        return self._bound_up

    @bound_up.setter
    def bound_up(self, bound):
        if isinstance(bound, (int, float)):
            bound = bound
        elif len(bound) == 1:
            bound = bound[0]
        else:
            bound = np.ravel(bound)
        self._bound_up = bound

    # def __call__(self, q, model):
    #     """
    #     The constraint function.
    #     """
    #     eqstate = model(q)
    #     return self.constraint(eqstate, model)

    def init(self, model):
        """
        Initialize the constraint with information from an equilibrium model.
        """
        self.index = self.index_from_model(model)

    @partial(jit, static_argnums=(0, 2))
    def __call__(self, q, model):
        """
        The constraint function.
        """
        eqstate = model(q)
        constraint = vmap(self.constraint, in_axes=(None, 0))(eqstate, self.index)
        return jnp.ravel(constraint)

    def constraint(self, eqstate, index):
        raise NotImplementedError
