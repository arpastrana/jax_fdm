"""
A collection of scipy-powered, gradient-based optimizers.
"""
from jax_fdm.optimization.optimizers import Optimizer
from jax_fdm.optimization.optimizers import ConstrainedOptimizer
from jax_fdm.optimization.optimizers import SecondOrderOptimizer


# ==========================================================================
# Optimizers
# ==========================================================================

class BFGS(Optimizer):
    """
    The Boyd-Fletcher-Floyd-Shannon optimizer.
    """
    def __init__(self, **kwargs):
        super().__init__(name="BFGS", **kwargs)


class LBFGSB(Optimizer):
    """
    The limited-memory Boyd-Fletcher-Floyd-Shannon-Byrd optimizer.
    """
    def __init__(self, **kwargs):
        super().__init__(name="L-BFGS-B", disp=0, **kwargs)


class TrustRegionConstrained(ConstrainedOptimizer):
    """
    A trust-region algorithm for constrained optimization.
    """
    def __init__(self, **kwargs):
        super().__init__(name="trust-constr", **kwargs)


class NewtonCG(SecondOrderOptimizer):
    """
    A trust-region algorithm for constrained optimization.
    """
    def __init__(self, **kwargs):
        super().__init__(name="Newton-CG", **kwargs)


class SLSQP(ConstrainedOptimizer):
    """
    The sequential least-squares programming optimizer.
    """
    def __init__(self, **kwargs):
        super().__init__(name="SLSQP", **kwargs)
