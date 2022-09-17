from functools import partial

import jax

from jax import jit

import jax.numpy as jnp


# ==========================================================================
# Loss
# ==========================================================================


class Loss:
    def __init__(self, *args, name=None):
        self.terms = args
        self.name = name or self.__class__.__name__

    @partial(jit, static_argnums=(0, 2))
    def __call__(self, q, model):
        eqstate = model(q)
        func = partial(self._term_val, eqstate=eqstate)
        return jnp.sum(jnp.array(jax.tree_map(func, self.terms)))

    @staticmethod
    @partial(jit, static_argnums=0)
    def _term_val(term, eqstate):
        return term(eqstate)
