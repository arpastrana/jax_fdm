import jax.numpy as jnp


class Regularizer:
    def __init__(self, alpha, name=None):
        self.alpha = alpha
        self.name = name or self.__class__.__name__


# TODO: Apply jnp.sqrt to sum of squares to make it a valid L2 norm
class L2Regularizer(Regularizer):
    def __call__(self, eqstate):
        return self.alpha * jnp.sum(jnp.square(eqstate.force_densities))
