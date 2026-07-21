from collections.abc import Callable
from functools import partial
from typing import Any

from jax import custom_vjp
from jax import vjp
from jaxopt.linear_solve import solve_normal_cg
from jaxtyping import Array
from jaxtyping import Float

from jax_fdm.equilibrium.solvers.types import SolverIterParams

# ==========================================================================
# Custom VJP via implicit differentiation
# ==========================================================================

__all__ = [
    "nonlinear_bwd",
    "nonlinear_fwd",
    "solver_nonlinear_implicit",
]


@partial(custom_vjp, nondiff_argnums=(0, 1, 2))
def solver_nonlinear_implicit(
    solver: Callable,
    solver_config: dict[str, Any],
    fn: Callable,
    theta: SolverIterParams,
    x_init: Float[Array, "nodes_free_flat"],
) -> Float[Array, "nodes_free_flat"]:
    """
    Solve a nonlinear system for ``x`` with implicit differentiation.

    Parameters
    ----------
    solver :
        The function that runs the nonlinear solve.
    solver_config :
        The configuration options of the solver.
    fn :
        The residual function to drive to zero.
    theta :
        The function parameters, differentiated through implicitly.
    x_init :
        The initial guess for the flattened solution vector.

    Returns
    -------
    x_star :
        The flattened solution vector.

    Notes
    -----
    Wrapped in a custom VJP so the backward pass differentiates through the
    solution implicitly rather than unrolling the solver iterations.
    """
    return solver(fn, theta, x_init, solver_config)


def nonlinear_fwd(
    solver: Callable,
    solver_config: dict[str, Any],
    fn: Callable,
    theta: SolverIterParams,
    x_init: Float[Array, "nodes_free_flat"],
) -> tuple[
    Float[Array, "nodes_free_flat"],
    tuple[SolverIterParams, Float[Array, "nodes_free_flat"]],
]:
    """
    Run the forward pass of the implicit nonlinear solver.

    Parameters
    ----------
    solver :
        The function that runs the nonlinear solve.
    solver_config :
        The configuration options of the solver.
    fn :
        The residual function to drive to zero.
    theta :
        The function parameters.
    x_init :
        The initial guess for the flattened solution vector.

    Returns
    -------
    result :
        The solution vector and the residual ``(theta, x_star)`` saved for the
        backward pass.
    """
    x_star = solver_nonlinear_implicit(solver, solver_config, fn, theta, x_init)

    # the custom_vjp wrapper's return type is opaque to pyright; x_star is a
    # jax.Array at runtime
    return x_star, (theta, x_star)  # pyright: ignore[reportReturnType]


def nonlinear_bwd(
    solver: Callable,
    solver_config: dict[str, Any],
    fn: Callable,
    res: tuple[SolverIterParams, Float[Array, "nodes_free_flat"]],
    vec: Float[Array, "nodes_free_flat"],
) -> tuple[SolverIterParams, None]:
    """
    Run the backward pass of the implicit nonlinear solver.

    Parameters
    ----------
    solver :
        The function that runs the nonlinear solve. Unused here.
    solver_config :
        The configuration options of the solver. Unused here.
    fn :
        The residual function driven to zero.
    res :
        The residual from the forward pass: the parameters and the solution.
    vec :
        The cotangent vector on the left of the VJP.

    Returns
    -------
    grads :
        The cotangent with respect to the parameters, and None for the unused
        initial guess.

    Notes
    -----
    The adjoint system is solved matrix-free with normal-equation conjugate
    gradients rather than by materializing the Jacobian.
    """
    theta, x_star = res

    # Solve adjoint system

    # A. Directly
    # jac_x_fn = jacfwd(fn, argnums=1)
    # Jx = jac_x_fn(theta, x_star)
    # lam = jnp.linalg.solve(Jx.T, -vec)

    # B. Iteratively
    _, vjp_x = vjp(lambda x: fn(theta, x).T, x_star)
    lam = solve_normal_cg(lambda w: vjp_x(w)[0], -vec)

    # Call vjp of residual_fn to compute gradient wrt parameters
    _, vjp_theta = vjp(lambda theta: fn(theta, x_star), theta)

    theta_bar = vjp_theta(lam)

    return theta_bar[0], None


# ==========================================================================
# Register custom VJP
# ==========================================================================

solver_nonlinear_implicit.defvjp(nonlinear_fwd, nonlinear_bwd)
