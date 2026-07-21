from collections.abc import Callable
from typing import Any

from jaxopt import GaussNewton
from jaxtyping import Array
from jaxtyping import Float
from optimistix import Dogleg
from optimistix import LevenbergMarquardt
from optimistix import least_squares

from jax_fdm.equilibrium.solvers.jaxopt import solver_jaxopt
from jax_fdm.equilibrium.solvers.optimistix import solver_optimistix
from jax_fdm.equilibrium.solvers.types import SolverIterParams

# ==========================================================================
# JAXOPT solvers
# ==========================================================================

__all__ = [
    "is_solver_leastsquares",
    "solver_dogleg",
    "solver_gauss_newton",
    "solver_levenberg_marquardt",
]


def solver_gauss_newton(
    fn: Callable,
    theta: SolverIterParams,
    x_init: Float[Array, "nodes_free_flat"],
    solver_config: dict[str, Any],
) -> Float[Array, "nodes_free_flat"]:
    """
    Drive the residual ``fn(theta, x)`` to zero with the Gauss-Newton algorithm.

    Parameters
    ----------
    fn :
        The residual function to drive to zero.
    theta :
        The function parameters.
    x_init :
        The initial guess for the flattened solution vector.
    solver_config :
        The configuration options of the solver.

    Returns
    -------
    x_star :
        The flattened solution vector at the residual minimum.
    """
    return solver_jaxopt(GaussNewton, fn, theta, x_init, solver_config)


# ==========================================================================
# Optimistix solvers
# ==========================================================================


def solver_levenberg_marquardt(
    fn: Callable,
    theta: SolverIterParams,
    x_init: Float[Array, "nodes_free_flat"],
    solver_config: dict[str, Any],
) -> Float[Array, "nodes_free_flat"]:
    """
    Drive the residual ``fn(theta, x)`` to zero with the Levenberg-Marquardt method.

    Parameters
    ----------
    fn :
        The residual function to drive to zero.
    theta :
        The function parameters.
    x_init :
        The initial guess for the flattened solution vector.
    solver_config :
        The configuration options of the solver.

    Returns
    -------
    x_star :
        The flattened solution vector at the residual minimum.

    Notes
    -----
    Incompatible with the sparse equilibrium model because
    ``jax.experimental.sparse.csr_matmat`` does not implement a batching rule yet.
    """
    solver_kwargs = {}

    solution = solver_optimistix(
        LevenbergMarquardt,
        least_squares,
        fn,
        theta,
        x_init,
        solver_config,
        solver_kwargs,
    )

    return solution


def solver_dogleg(
    fn: Callable,
    theta: SolverIterParams,
    x_init: Float[Array, "nodes_free_flat"],
    solver_config: dict[str, Any],
) -> Float[Array, "nodes_free_flat"]:
    """
    Drive the residual ``fn(theta, x)`` to zero with the Dogleg trust-region method.

    Parameters
    ----------
    fn :
        The residual function to drive to zero.
    theta :
        The function parameters.
    x_init :
        The initial guess for the flattened solution vector.
    solver_config :
        The configuration options of the solver.

    Returns
    -------
    x_star :
        The flattened solution vector at the residual minimum.
    """
    solver_kwargs = {}

    solution = solver_optimistix(
        Dogleg,
        least_squares,
        fn,
        theta,
        x_init,
        solver_config,
        solver_kwargs,
    )

    return solution


# ==========================================================================
# Helper functions
# ==========================================================================


def is_solver_leastsquares(solver_fn: Callable) -> bool:
    """
    Test whether a solver function is a least-squares solver.

    Parameters
    ----------
    solver_fn :
        The solver function to test.

    Returns
    -------
    is_leastsquares :
        True if the solver is a least-squares solver, otherwise False.
    """
    solver_fns = {solver_gauss_newton, solver_levenberg_marquardt, solver_dogleg}

    return solver_fn in solver_fns
