from jax import jit
from jax import jacfwd
from jax import jacrev

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
        # NOTE: jacrev(jacfwd) is x3 slower than hessian. Why?
        # NOTE: Ah, but jacfwd(jacrev) is as fast as hessian
        return jit(jacfwd(jacrev(loss, argnums=0)))
