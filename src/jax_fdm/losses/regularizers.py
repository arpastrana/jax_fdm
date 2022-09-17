import jax.numpy as jnp


class Regularizer:
    def __init__(self, alpha, name=None):
        self.alpha = alpha
        self.name = name or self.__class__.__name__


class L2Regularizer(Regularizer):
    def __call__(self, eqstate):
        return self.alpha * jnp.sum(jnp.square(eqstate.force_densities))
