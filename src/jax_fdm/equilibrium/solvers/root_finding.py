try:
    from optimistix import Newton
    from optimistix import root_find
    from lineax import SVD

except ImportError:
    pass

from jax_fdm.equilibrium.solvers import solver_optimistix


# ==========================================================================
# Optimistix solvers
# ==========================================================================

def solver_newton(fn, theta, x_init, solver_config):
    """
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
        solver_kwargs
    )

    return solution


# ==========================================================================
# Helper functions
# ==========================================================================

def is_solver_root_finding(solver):
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
