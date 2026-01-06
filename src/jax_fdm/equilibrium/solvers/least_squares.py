from jaxopt import GaussNewton

from jax_fdm.equilibrium.solvers.jaxopt import solver_jaxopt
from jax_fdm.equilibrium.solvers.optimistix import solver_optimistix

try:
    from optimistix import Dogleg
    from optimistix import LevenbergMarquardt
    from optimistix import least_squares

except ImportError:
    pass


# ==========================================================================
# JAXOPT solvers
# ==========================================================================

def solver_gauss_newton(fn, theta, x_init, solver_config):
    """
    Minimize the residual of function f(theta, x) = 0 using the Gauss Newton algorithm.

    Parameters
    ----------
    f : The function to iterate upon.
    a : The function parameters.
    x_init: An initial guess for the values of the solution vector.
    solver_config: The configuration options of the solver.

    Returns
    -------
    x_star : The solution vector at the fixed point.
    """
    return solver_jaxopt(GaussNewton, fn, theta, x_init, solver_config)


# ==========================================================================
# Optimistix solvers
# ==========================================================================

def solver_levenberg_marquardt(fn, theta, x_init, solver_config):
    """
    Minimize the residual of function f(theta, x) = 0 using the Levenberg Marquardt algorithm.

    Parameters
    ----------
    f : The function to iterate upon.
    a : The function parameters.
    x_init: An initial guess for the values of the solution vector.
    solver_config: The configuration options of the solver.

    Returns
    -------
    x_star : The solution vector at the fixed point.

    Notes
    -----
    This solver is incompatible with `EquilibriumModelSparse` because
    `jax.experimental.sparse.csr_matmat` does not implement a batching rule yet.
    """
    solver_kwargs = {}

    solution = solver_optimistix(
        LevenbergMarquardt,
        least_squares,
        fn,
        theta,
        x_init,
        solver_config,
        solver_kwargs
    )

    return solution


def solver_dogleg(fn, theta, x_init, solver_config):
    """
    """
    solver_kwargs = {}

    solution = solver_optimistix(
        Dogleg,
        least_squares,
        fn,
        theta,
        x_init,
        solver_config,
        solver_kwargs
    )

    return solution


# ==========================================================================
# Helper functions
# ==========================================================================

def is_solver_leastsquares(solver_fn):
    """
    Test if a solver function is a least squares solver.

    Parameters
    ----------
    `solver_fn`: A solver function

    Returns
    -------
    `True` if the solver is a least squares solver. Otherwise, `False`.
    """
    solver_fns = {
        solver_gauss_newton,
        solver_levenberg_marquardt,
        solver_dogleg
    }

    return solver_fn in solver_fns
