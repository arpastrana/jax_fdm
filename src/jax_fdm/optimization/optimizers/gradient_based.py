"""SciPy-backed gradient-based optimizers."""

from collections.abc import Callable
from typing import Any

import numpy as np

from jax_fdm.optimization.optimizers.constrained import ConstrainedOptimizer
from jax_fdm.optimization.optimizers.optimizer import Optimizer
from jax_fdm.optimization.optimizers.second_order import SecondOrderOptimizer

# ==========================================================================
# Optimizers
# ==========================================================================

__all__ = [
    "BFGS",
    "LBFGSB",
    "LBFGSBS",
    "SLSQP",
    "NewtonCG",
    "TruncatedNewton",
    "TrustRegionConstrained",
    "TrustRegionExact",
    "TrustRegionKrylov",
    "TrustRegionNewton",
]


class SLSQP(ConstrainedOptimizer):
    """
    The sequential least-squares programming optimizer.
    """

    name = "SLSQP"


class LBFGSB(Optimizer):
    """
    The limited-memory BFGS optimizer with box bounds (L-BFGS-B).

    Parameters
    ----------
    disp :
        Whether the SciPy backend prints its own console output.
    maxfun :
        The maximum number of function evaluations. If None, the backend default.
    maxls :
        The maximum number of line-search steps per iteration. If None, the
        backend default.
    maxcor :
        The number of correction pairs approximating the hessian. If None, the
        backend default.
    """

    name = "L-BFGS-B"

    def __init__(
        self,
        disp: bool = False,
        maxfun: int | None = None,
        maxls: int | None = None,
        maxcor: int | None = None,
        **kwargs: Any,
    ):
        super().__init__(disp=disp, **kwargs)
        self.maxfun = maxfun
        self.maxls = maxls
        self.maxcor = maxcor

    def options(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Assemble the backend options, adding the L-BFGS-B specific limits.

        Parameters
        ----------
        extra :
            Extra options to merge in. None-valued entries are skipped.

        Returns
        -------
        options :
            The options dictionary with ``maxfun``, ``maxls``, and ``maxcor`` added
            and ``disp`` removed, as SciPy >= 1.16 no longer accepts it here.
        """
        if extra is None:
            extra = {}

        extra["maxfun"] = self.maxfun
        extra["maxls"] = self.maxls
        extra["maxcor"] = self.maxcor

        options = super().options(extra)

        # scipy>=1.16 dropped `disp` from L-BFGS-B's accepted options (and it has
        # no options-based verbosity control left), so passing it warns.
        options.pop("disp", None)

        return options


class LBFGSBS(LBFGSB):
    """
    An L-BFGS-B variant that converts JAX gradients to NumPy for SciPy.

    Notes
    -----
    Wrapping each gradient in a NumPy array sidesteps SciPy compatibility issues
    with JAX arrays, at the cost of being slower than
    [LBFGSB][jax_fdm.optimization.optimizers.gradient_based.LBFGSB].
    """

    def gradient(self, loss: Callable) -> Callable:
        """
        Build the gradient, returning NumPy arrays instead of JAX arrays.

        Parameters
        ----------
        loss :
            The loss function to differentiate.

        Returns
        -------
        gradient :
            The jitted gradient wrapped to return a NumPy array.
        """
        jit_gfunc = super().gradient(loss)
        return lambda x: np.array(jit_gfunc(x))


class BFGS(Optimizer):
    """
    The Boyd-Fletcher-Floyd-Shannon optimizer.
    """

    name = "BFGS"


class NewtonCG(SecondOrderOptimizer):
    """
    The truncated Newton method. It uses a CG method to the compute the search
    direction.
    """

    name = "Newton-CG"


class TruncatedNewton(Optimizer):
    """
    Minimize a scalar function of one or more variables using a truncated Newton
    (TNC) algorithm.
    """

    name = "TNC"


class TrustRegionConstrained(ConstrainedOptimizer):
    """
    A trust-region algorithm for constrained optimization.
    """

    name = "trust-constr"


class TrustRegionKrylov(SecondOrderOptimizer):
    """
    A trust-region optimization algorithm.

    It uses a nearly exact trust-region algorithm that only requires
    matrix vector products with the hessian matrix.
    """

    name = "trust-krylov"


class TrustRegionNewton(SecondOrderOptimizer):
    """
    A Newton conjugate gradient trust-region algorithm.
    A trust-region algorithm for unconstrained optimization.
    """

    name = "trust-ncg"


class TrustRegionExact(SecondOrderOptimizer):
    """
    A nearly exact trust-region optimization algorithm.
    """

    name = "trust-exact"
