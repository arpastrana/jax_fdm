from collections.abc import Callable

from jax import jacfwd
from jax import jacrev

from jax_fdm.optimization.optimizers.optimizer import Optimizer

# ==========================================================================
# Second-order optimizer
# ==========================================================================


class SecondOrderOptimizer(Optimizer):
    """
    A gradient-based optimizer that uses the hessian to accelerate convergence.
    """

    def hessian(self, loss: Callable) -> Callable:
        """
        Build the hessian of a loss function by nested autodiff.

        Parameters
        ----------
        loss :
            The loss function to differentiate twice.

        Returns
        -------
        hessian :
            The hessian with respect to the optimization parameters.

        Notes
        -----
        Composed as ``jacfwd(jacrev(...))``, which matches the speed of a dedicated
        hessian, whereas ``jacrev(jacfwd(...))`` is about three times slower.
        """
        return jacfwd(jacrev(loss, argnums=0))
