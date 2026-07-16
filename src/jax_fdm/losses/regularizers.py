import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium import EquilibriumParametersState


class Regularizer:
    """
    A regularizer is a function that penalizes the parameters of a model.
    """

    def __init__(self, alpha: float, name: str | None = None) -> None:
        self.alpha = alpha
        self.name = name or self.__class__.__name__

    def __call__(self, params: EquilibriumParametersState) -> Float[Array, ""]:
        """
        The regularization value.
        """
        raise NotImplementedError("Regularizer subclasses must implement __call__")


class L2Regularizer(Regularizer):
    """
    A regularizer that penalizes the L2 norm of the force densities.
    """

    def __call__(self, params: EquilibriumParametersState) -> Float[Array, ""]:
        """
        The regularization value.
        """
        # TODO: Apply jnp.sqrt to sum of squares to make it a valid L2 norm?
        return self.alpha * jnp.sum(jnp.square(params.q))
