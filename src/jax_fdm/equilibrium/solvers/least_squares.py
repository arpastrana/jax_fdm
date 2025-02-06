from functools import partial

import jax.numpy as jnp

from jax import custom_vjp
from jax import jacfwd
from jax import vjp

from jaxopt import GaussNewton
from jaxopt import LevenbergMarquardt
from jaxopt import LBFGS
from jaxopt import ScipyMinimize

from jax_fdm.equilibrium.solvers.jaxopt import solver_jaxopt

from jax_fdm.equilibrium.solvers.optimistix import solver_levenberg_marquardt_optimistix
from jax_fdm.equilibrium.solvers.optimistix import solver_dogleg_optimistix

# ==========================================================================
# Iterative solvers - JAXOPT
# ==========================================================================

def solver_gauss_newton(f, solver_config):
    """
    Minimize the residual of f(x, theta) = 0 with the Gauss Newton algorithm.

    Parameters
    ----------
    f : The function to iterate upon.
    solver_config: The configuration options of the solver.

    Returns
    -------
    x_star : The solution vector.
    """
    return solver_jaxopt(GaussNewton, f, solver_config)


def solver_levenberg_marquardt(f, solver_config):
    """
    Minimize the residual of f(x, theta) = 0 with the Levenberg Marquardt algorithm.

    Parameters
    ----------
    f : The function to iterate upon.
    solver_config: The configuration options of the solver.

    Returns
    -------
    x_star : The solution vector.

    Notes
    -----
    This solver is incompatible with `EquilibriumModelSparse` because
    `jax.experimental.sparse.csr_matmat` does not implement a batching rule yet.
    """
    solver_kwargs = {"solver": "lu", "materialize_jac": True}
    # solver_kwargs = {}
    return solver_jaxopt(LevenbergMarquardt, f, solver_config, solver_kwargs)


def solver_lbfgs(f, solver_config):
    """
    Minimize the residual of f(x, theta) = 0 with the LBFGS algorithm.

    Parameters
    ----------
    f : The function to iterate upon.
    solver_config: The configuration options of the solver.

    Returns
    -------
    x_star : The solution vector.
    """
    return solver_jaxopt(LBFGS, f, solver_config)


def solver_lbfgs_scipy(f, solver_config):
    """
    Minimize the residual of f(x, theta) = 0 with scipy's LBFGS algorithm.

    Parameters
    ----------
    f : The function to iterate upon.
    solver_config: The configuration options of the solver.

    Returns
    -------
    x_star : The solution vector.
    """
    solver_kwargs = {"method": "L-BFGS-B"}

    return solver_jaxopt(ScipyMinimize, f, solver_config, solver_kwargs)


def is_solver_leastsquares(solver):
    """
    Test if a solver function is a least squares solver.

    Parameters
    ----------
    `solver_fn`: A solver function

    Returns
    -------
    `True` if the solver is a least squares solver. Otherwise, `False`.
    """
    solvers = {
        solver_gauss_newton,
        solver_levenberg_marquardt,
        solver_lbfgs,
        solver_lbfgs_scipy,
        solver_levenberg_marquardt_optimistix,
        solver_dogleg_optimistix,
    }

    return solver in solvers

# ==========================================================================
# Fixed point solver wrapper for implicit differentiation
# ==========================================================================

@partial(custom_vjp, nondiff_argnums=(0, ))
def least_squares(solver, x_init, theta, structure):
    """
    Find a minimum of f(x, theta) in a least-squares sense using an iterative solver.
    """
    return solver(x_init=x_init, theta=theta, structure=structure)


def least_squares_fwd(solver, x_init, theta, structure):
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
    x_star = least_squares(solver, x_init, theta, structure)

    return x_star, (x_star, theta, structure)


def least_squares_bwd(solver, res, vec):
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
    # Fetch residual function from solver
    f = solver.keywords["fn"]

    # Unpack data from the forward pass
    x_star, theta, structure = res

    # Solve adjoint system
    # _, vjp_x = vjp(lambda x: fn(theta, x), x_star)
    # _ = vjp_x(-vec)

    jac_fn = jacfwd(f, argnums=0)

    J = jac_fn(x_star, theta, structure)
    lam = jnp.linalg.solve(J.T, -vec)

    # Call vjp of residual_fn to compute gradient wrt parameters
    _, vjp_theta = vjp(lambda theta: f(x_star, theta, structure), theta)

    theta_bar = vjp_theta(lam)

    return None, theta_bar[0], None


# ==========================================================================
# Register custom VJP
# ==========================================================================

least_squares.defvjp(least_squares_fwd, least_squares_bwd)
