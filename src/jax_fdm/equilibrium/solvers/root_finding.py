from collections.abc import Callable
from typing import Any

from jaxtyping import Array
from jaxtyping import Float
from lineax import SVD
from optimistix import Newton
from optimistix import root_find

from jax_fdm.equilibrium.solvers import solver_optimistix
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
    Find a root of function f(theta, x) = 0 using Newton's method.
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
    Tests if a solver function is a root finding solver.

    Parameters
    ----------
    `solver_fn`: A solver function

    Returns
    -------
    `True` if the solver is a least squares solver. Otherwise, `False`.
    """
    solvers = {
        solver_newton,
    }

    return solver in solvers
