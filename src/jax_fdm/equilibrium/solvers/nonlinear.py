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


@partial(custom_vjp, nondiff_argnums=(0, 1, 2))
def solver_nonlinear_implicit(
    solver: Callable,
    solver_config: dict[str, Any],
    fn: Callable,
    theta: SolverIterParams,
    x_init: Float[Array, "nodes_free_flat"],
) -> Float[Array, "nodes_free_flat"]:
    """
    Find a minimum of f(theta, x) in a least-squares sense using an iterative solver.
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
