import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium import EquilibriumParametersState


class Regularizer:
    def __init__(self, alpha: float, name: str | None = None):
        self.alpha = alpha
        self.name = name or self.__class__.__name__


class L2Regularizer(Regularizer):
    def __call__(self, params: EquilibriumParametersState) -> Float[Array, ""]:
        # TODO: Apply jnp.sqrt to sum of squares to make it a valid L2 norm?
        return self.alpha * jnp.sum(jnp.square(params.q))
