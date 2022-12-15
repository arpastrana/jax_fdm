"""
A collection of scipy-powered, gradient-based optimizers.
"""
import numpy as np

from jax_fdm.optimization.optimizers import Optimizer
from jax_fdm.optimization.optimizers import ConstrainedOptimizer
from jax_fdm.optimization.optimizers import SecondOrderOptimizer


# ==========================================================================
# Optimizers
# ==========================================================================

class SLSQP(ConstrainedOptimizer):
    """
    The sequential least-squares programming optimizer.
    """
    def __init__(self, **kwargs):
        super().__init__(name="SLSQP", **kwargs)


class LBFGSB(Optimizer):
    """
    The limited-memory Boyd-Fletcher-Floyd-Shannon-Byrd (LBFGSB) optimizer.
    """
    def __init__(self, **kwargs):
        super().__init__(name="L-BFGS-B", disp=0, **kwargs)


class LBFGSBS(Optimizer):
    """
    The limited-memory Boyd-Fletcher-Floyd-Shannon-Byrd (LBFGSB) optimizer.

    This version of LBFGSB ensures compatibility of JAX gradients with scipy.
    However, it is slower than standard LBFGSB.
    """
    def __init__(self, **kwargs):
        super().__init__(name="L-BFGS-B", disp=0, **kwargs)

    def gradient(self, loss):
        """
        Compute the gradient function of a loss function.
        """
        jit_gfunc = super().gradient(loss)
        return lambda x: np.array(jit_gfunc(x))


class BFGS(Optimizer):
    """
    The Boyd-Fletcher-Floyd-Shannon optimizer.
    """
    def __init__(self, **kwargs):
        super().__init__(name="BFGS", **kwargs)


class TrustRegionConstrained(ConstrainedOptimizer, SecondOrderOptimizer):
    """
    A trust-region algorithm for constrained optimization.
    """
    def __init__(self, **kwargs):
        super().__init__(name="trust-constr", **kwargs)


class NewtonCG(SecondOrderOptimizer):
    """
    The truncated Newton method. It uses a CG method to the compute the search direction.
    """
    def __init__(self, **kwargs):
        super().__init__(name="Newton-CG", **kwargs)
