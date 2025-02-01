from functools import partial

import jax.numpy as jnp

from jax import custom_vjp
from jax import jacrev
from jax import vjp

from jaxopt import GaussNewton
from jaxopt import LevenbergMarquardt

from jax_fdm.equilibrium.solvers.jaxopt import solver_jaxopt


# ==========================================================================
# Iterative solvers - JAXOPT
# ==========================================================================

def solver_gauss_newton(f, a, x_init, solver_config):
    """
    Minimize the residual of function f(a, x) = 0 using the Gauss Newton algorithm.

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
    return solver_jaxopt(GaussNewton, f, a, x_init, solver_config)


def solver_levenberg_marquardt(f, a, x_init, solver_config):
    """
    Minimize the residual of function f(a, x) = 0 using the Levenberg Marquardt algorithm.

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
    return solver_jaxopt(LevenbergMarquardt, f, a, x_init, solver_config)


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
    return solver_fn in {solver_gauss_newton, solver_levenberg_marquardt}

# ==========================================================================
# Fixed point solver wrapper for implicit differentiation
# ==========================================================================

@partial(custom_vjp, nondiff_argnums=(0, 1, 2))
def least_squares(solver, solver_config, fn, theta, x_init):
    """
    Find a minimum of f(theta, x) in a least-squares sense using an iterative solver.
    """
    return solver(fn, theta, x_init, solver_config)


def least_squares_fwd(solver, solver_config, fn, theta, x_init):
    """
    The forward pass of an iterative least squares solver.

    Parameters
    ----------
    solver: The function that executes a least_squares solver.
    solver_config: The configuration options of the solver.
    fn : The function to iterate upon.
    theta : The function parameters.
    x_init: An initial guess for the values of the solution vector.

    Returns
    -------
    x_star : The solution vector at a fixed point.
    res : Auxiliary data to transfer to the backward pass.
    """
    x_star = least_squares(solver, solver_config, fn, theta, x_init)

    return x_star, (theta, x_star)


def least_squares_bwd(solver, solver_config, fn, res, vec):
    """
    The backward pass of an iterative least squares solver.

    Parameters
    ----------
    solver: The function that executes a fixed point solver.
    solver_config: The configuration options of the solver.
    fn : The function to iterate upon.
    res : Auxiliary data transferred from the forward pass.
    vec: The vector on the left of the VJP.

    Returns
    -------
    theta_bar: the VJP vector of fn w.r.t. the parameters `theta`
    """
    theta, x_star = res
    pass


# ==========================================================================
# Register custom VJP
# ==========================================================================

least_squares.defvjp(least_squares_fwd, least_squares_bwd)
