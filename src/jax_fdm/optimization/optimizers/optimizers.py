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


class NewtonCG(SecondOrderOptimizer):
    """
    The truncated Newton method. It uses a CG method to the compute the search direction.
    """
    def __init__(self, **kwargs):
        super().__init__(name="Newton-CG", **kwargs)


class TruncatedNewton(Optimizer):
    """
    Minimize a scalar function of one or more variables using a truncated Newton (TNC) algorithm.
    """
    def __init__(self, **kwargs):
        super().__init__(name="TNC", **kwargs)


class TrustRegionConstrained(ConstrainedOptimizer):
    """
    A trust-region algorithm for constrained optimization.
    """
    def __init__(self, **kwargs):
        super().__init__(name="trust-constr", **kwargs)


class TrustRegionKrylov(SecondOrderOptimizer):
    """
    A trust-region optimization algorithm.

    It uses a nearly exact trust-region algorithm that only requires
    matrix vector products with the hessian matrix.
    """
    def __init__(self, **kwargs):
        super().__init__(name="trust-krylov", **kwargs)


class TrustRegionNewton(SecondOrderOptimizer):
    """
    A Newton conjugate gradient trust-region algorithm.
    A trust-region algorithm for unconstrained optimization.
    """
    def __init__(self, **kwargs):
        super().__init__(name="trust-ncg", **kwargs)


class TrustRegionExact(SecondOrderOptimizer):
    """
    A nearly exact trust-region optimization algorithm.
    """
    def __init__(self, **kwargs):
        super().__init__(name="trust-exact", **kwargs)
