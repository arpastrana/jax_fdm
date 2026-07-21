from collections.abc import Callable
from typing import Any

from jaxtyping import Array
from jaxtyping import Float
from lineax import SVD
from optimistix import Newton
from optimistix import root_find

from jax_fdm.equilibrium.solvers.optimistix import solver_optimistix
from jax_fdm.equilibrium.solvers.types import SolverIterParams

# ==========================================================================
# Optimistix solvers
# ==========================================================================


def solver_newton(
    fn: Callable,
    theta: SolverIterParams,
    x_init: Float[Array, "nodes_free_flat"],
    solver_config: dict[str, Any],
) -> Float[Array, "nodes_free_flat"]:
    """
    Find a root of ``fn(theta, x) = 0`` using Newton's method.

    Parameters
    ----------
    fn :
        The residual function whose root is sought.
    theta :
        The function parameters.
    x_init :
        The initial guess for the flattened solution vector.
    solver_config :
        The configuration options of the solver.

    Returns
    -------
    x_star :
        The flattened solution vector at the root.

    Notes
    -----
    Uses an SVD linear solver inside Newton's method to stay robust when the
    Jacobian is singular or ill-conditioned.
    """
    solver_config["verbose"] = False
    solver_kwargs = {"linear_solver": SVD()}

    solution = solver_optimistix(
        Newton,
        root_find,
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


def is_solver_root_finding(solver: Callable) -> bool:
    """
    Test whether a solver function is a root-finding solver.

    Parameters
    ----------
    solver :
        The solver function to test.

    Returns
    -------
    is_root_finding :
        True if the solver is a root-finding solver, otherwise False.
    """
    solvers = {
        solver_newton,
    }

    return solver in solvers
