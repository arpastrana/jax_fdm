import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium import EquilibriumParametersState

__all__ = [
    "L2Regularizer",
    "Regularizer",
]


class Regularizer:
    """
    The base class for a loss term that penalizes the model parameters.

    Parameters
    ----------
    alpha :
        A scalar weight scaling the regularization term in the loss.
    name :
        The name of the regularizer. If None, defaults to the class name.
    """

    def __init__(self, alpha: float, name: str | None = None) -> None:
        self.alpha = alpha
        self.name = name or self.__class__.__name__

    def __call__(self, params: EquilibriumParametersState) -> Float[Array, ""]:
        """
        Evaluate the regularization penalty on the parameters.

        Parameters
        ----------
        params :
            The parameter state to penalize.

        Returns
        -------
        penalty :
            The scalar regularization penalty.
        """
        raise NotImplementedError("Regularizer subclasses must implement __call__")


class L2Regularizer(Regularizer):
    """
    A penalty on the squared force densities.

    Notes
    -----
    Computes ``alpha`` times the sum of squared force densities, the squared L2
    norm rather than the norm itself.
    """

    def __call__(self, params: EquilibriumParametersState) -> Float[Array, ""]:
        """
        The weighted sum of squared force densities.

        Parameters
        ----------
        params :
            The parameter state whose force densities are penalized.

        Returns
        -------
        penalty :
            ``alpha`` times the sum of squared force densities.
        """
        return self.alpha * jnp.sum(jnp.square(params.q))
