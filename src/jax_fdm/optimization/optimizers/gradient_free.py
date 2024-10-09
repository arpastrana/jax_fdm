"""
A collection of scipy-powered, gradient-free optimizers.
"""
from jax_fdm.optimization.optimizers import Optimizer


# ==========================================================================
# Optimizers
# ==========================================================================

class Powell(Optimizer):
    """
    The modified Powell algorithm for gradient-free optimization with box constraints.
    """
    def __init__(self, **kwargs):
        super().__init__(name="Powell", disp=0, **kwargs)


class NelderMead(Optimizer):
    """
    The Nelder-Mead gradient-free optimizer with box constraints.
    """
    def __init__(self, **kwargs):
        super().__init__(name="Nelder-Mead", disp=0, **kwargs)
