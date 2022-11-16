from jax import jit
from jax import hessian

from jax_fdm.optimization.optimizers import Optimizer


# ==========================================================================
# Second-order optimizer
# ==========================================================================

class SecondOrderOptimizer(Optimizer):
    """
    A gradient-based optimizer that uses the hessian to accelerate convergence.
    """
    def hessian(self, loss):
        """
        Compute the hessian function of a loss function.
        """
        return jit(hessian(loss, argnums=0))
