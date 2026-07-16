"""
Regression tests for the standalone gradient descent solver.
"""

import numpy as np

# Import a top-level subpackage first to fully initialize jax_fdm before
# reaching into the optimizer module, sidestepping the optimization/equilibrium
# import cycle.
import jax_fdm.equilibrium  # noqa: F401
from jax_fdm.optimization.optimizers.gradient_descent import gradient_descent

MAXITER = 3


def _quadratic(x):
    """
    A simple convex objective with its minimum at the origin.
    """
    return float(np.sum(x**2))


def test_gradient_descent_finite_difference_runs():
    """
    Without an explicit jacobian, the finite-difference path runs and descends.
    """
    x0 = np.array([1.0, 2.0])
    result = gradient_descent(
        _quadratic,
        x0,
        jac=None,
        tol=1e-6,
        options={"maxiter": MAXITER},
    )

    assert result.fun < _quadratic(x0)
    assert np.all(np.abs(result.x) < np.abs(x0))


def test_gradient_descent_finite_difference_counts_nfev():
    """
    The finite-difference path counts every function evaluation in nfev.
    """
    x0 = np.array([1.0, 2.0])
    result = gradient_descent(
        _quadratic,
        x0,
        jac=None,
        tol=1e-6,
        options={"maxiter": MAXITER},
    )

    # Each iteration evaluates the objective once for f0 and once per dimension
    # inside approx_fprime, so the count grows past the iteration count.
    assert result.nfev > MAXITER
