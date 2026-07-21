from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from equinox.internal import while_loop
from jax import custom_vjp
from jax import jacfwd
from jax import vjp
from jax.experimental.sparse import JAXSparse
from jax.lax import custom_linear_solve
from jax.scipy.sparse.linalg import bicgstab
from jaxopt import AndersonAcceleration
from jaxopt import FixedPointIteration
from jaxtyping import Array
from jaxtyping import Float
from lineax import CG
from lineax import FunctionLinearOperator
from lineax import Normal
from lineax import linear_solve

from jax_fdm.equilibrium.solvers.jaxopt import solver_jaxopt
from jax_fdm.equilibrium.solvers.types import SolverIterParams
from jax_fdm.equilibrium.sparse import splu_cpu as splu
from jax_fdm.equilibrium.sparse import splu_solve_cpu as splu_solve

# ==========================================================================
# Iterative solvers - JAXOPT
# ==========================================================================

__all__ = [
    "fixed_point_bwd_adjoint",
    "fixed_point_bwd_adjoint_general",
    "fixed_point_bwd_fixedpoint",
    "fixed_point_bwd_materialize",
    "fixed_point_fwd",
    "is_solver_fixedpoint",
    "solver_anderson",
    "solver_fixedpoint",
    "solver_fixedpoint_implicit",
    "solver_forward",
]


def solver_anderson(
    f: Callable,
    a: SolverIterParams,
    x_init: Float[Array, "nodes_free 3"],
    solver_config: dict[str, Any],
) -> Float[Array, "nodes_free 3"]:
    """
    Find a fixed point of ``f(a, x)`` using Anderson acceleration.

    Parameters
    ----------
    f :
        The function to iterate upon.
    a :
        The function parameters.
    x_init :
        The initial guess for the solution vector.
    solver_config :
        The configuration options of the solver.

    Returns
    -------
    x_star :
        The solution vector at the fixed point.
    """
    solver_kwargs = {"history_size": 5, "ridge": 1e-6}

    return solver_jaxopt(
        AndersonAcceleration,
        f,
        a,
        x_init,
        solver_config,
        solver_kwargs,
    )


def solver_fixedpoint(
    f: Callable,
    a: SolverIterParams,
    x_init: Float[Array, "nodes_free 3"],
    solver_config: dict[str, Any],
) -> Float[Array, "nodes_free 3"]:
    """
    Find a fixed point of ``f(a, x)`` using plain fixed-point iteration.

    Parameters
    ----------
    f :
        The function to iterate upon.
    a :
        The function parameters.
    x_init :
        The initial guess for the solution vector.
    solver_config :
        The configuration options of the solver.

    Returns
    -------
    x_star :
        The solution vector at the fixed point.
    """
    return solver_jaxopt(FixedPointIteration, f, a, x_init, solver_config)


def is_solver_fixedpoint(solver_fn: Callable) -> bool:
    """
    Test whether a solver function is a fixed-point solver.

    Parameters
    ----------
    solver_fn :
        The solver function to test.

    Returns
    -------
    is_fixedpoint :
        True if the solver is a fixed-point solver, otherwise False.
    """
    solver_fns = {
        solver_anderson,
        solver_fixedpoint,
        solver_forward,
    }

    return solver_fn in solver_fns


# ==========================================================================
# Homecooked solvers
# ==========================================================================


def solver_forward(
    f: Callable,
    a: Any,
    x_init: Float[Array, "..."],
    solver_config: dict[str, Any],
) -> Float[Array, "..."]:
    """
    Find a fixed point of ``f(a, x)`` by forward iteration until convergence.

    Parameters
    ----------
    f :
        The function to iterate upon.
    a :
        The function parameters.
    x_init :
        The initial guess for the solution vector.
    solver_config :
        The configuration options of the solver, read for ``tmax``, ``eta``, and
        ``verbose``.

    Returns
    -------
    x_star :
        The solution vector at the fixed point.

    Notes
    -----
    Iteration stops when the mean nodal move between successive iterates drops
    below ``eta`` or after ``tmax`` steps, whichever comes first.
    """
    tmax = solver_config["tmax"]
    eta = solver_config["eta"]
    verbose = solver_config["verbose"]

    def distance(x_prev, x):
        residual = jnp.mean(jnp.linalg.norm(x_prev - x, axis=1))
        if verbose:
            jax.debug.print("Residual: {}", residual)
        return residual

    def cond_fun(carry):
        x_prev, x = carry
        return distance(x_prev, x) > eta

    def body_fun(carry):
        _, x = carry
        return x, f(a, x)

    init_val = (x_init, f(a, x_init))

    _, x_star = while_loop(
        cond_fun,
        body_fun,
        init_val,
        max_steps=tmax,
        kind="checkpointed",
    )

    return x_star


# ==========================================================================
# Fixed point solver wrapper for implicit differentiation
# ==========================================================================


@partial(custom_vjp, nondiff_argnums=(0, 1, 2))
def solver_fixedpoint_implicit(
    solver: Callable,
    solver_config: dict[str, Any],
    f: Callable,
    a: SolverIterParams,
    x_init: Float[Array, "nodes_free 3"],
) -> Float[Array, "nodes_free 3"]:
    """
    Solve for a fixed point of ``f(a, x)`` with implicit differentiation.

    Parameters
    ----------
    solver :
        The function that runs the fixed-point solve.
    solver_config :
        The configuration options of the solver.
    f :
        The function to iterate upon.
    a :
        The function parameters, differentiated through implicitly.
    x_init :
        The initial guess for the solution vector.

    Returns
    -------
    x_star :
        The solution vector at the fixed point.

    Notes
    -----
    Wrapped in a custom VJP so the backward pass differentiates through the fixed
    point implicitly rather than unrolling the solver iterations.
    """
    return solver(f, a, x_init, solver_config)


def fixed_point_fwd(
    solver: Callable,
    solver_config: dict[str, Any],
    f: Callable,
    a: SolverIterParams,
    x_init: Float[Array, "nodes_free 3"],
) -> tuple[
    Float[Array, "nodes_free 3"],
    tuple[SolverIterParams, Float[Array, "nodes_free 3"]],
]:
    """
    Run the forward pass of the implicit fixed-point solver.

    Parameters
    ----------
    solver :
        The function that runs the fixed-point solve.
    solver_config :
        The configuration options of the solver.
    f :
        The function to iterate upon.
    a :
        The function parameters.
    x_init :
        The initial guess for the solution vector.

    Returns
    -------
    result :
        The solution vector and the residual ``(a, x_star)`` saved for the backward
        pass.
    """
    x_star = solver_fixedpoint_implicit(solver, solver_config, f, a, x_init)

    # the custom_vjp wrapper's return type is opaque to pyright; x_star is a
    # jax.Array at runtime
    return x_star, (a, x_star)  # pyright: ignore[reportReturnType]


# ==========================================================================
# Backward rule - Materialize Jacobian
# ==========================================================================


def fixed_point_bwd_materialize(
    solver: Callable,
    solver_config: dict[str, Any],
    f: Callable,
    res: tuple[SolverIterParams, Float[Array, "nodes_free 3"]],
    vec: Float[Array, "nodes_free 3"],
) -> tuple[SolverIterParams, None]:
    """
    Run the backward pass by materializing the fixed-point Jacobian densely.

    Parameters
    ----------
    solver :
        The function that runs the fixed-point solve. Unused here.
    solver_config :
        The configuration options of the solver. Unused here.
    f :
        The function iterated upon, with signature ``f(theta, x(theta))``.
    res :
        The residual from the forward pass: the parameters and the fixed point.
    vec :
        The cotangent vector on the left of the VJP.

    Returns
    -------
    grads :
        The cotangent with respect to the parameters, and None for the unused
        initial guess.

    Notes
    -----
    Forms the dense adjoint matrix ``I - J`` and solves it directly. Simple but
    memory-heavy; the adjoint and iterative variants avoid the dense solve.
    """
    # Unpack data from forward pass
    theta, x_star = res

    # Format data
    x_star_flat = x_star.ravel()

    def f_ravel(theta, x):
        x = x.reshape(-1, 3)
        return f(theta, x).ravel()

    # NOTE: Use jacrev or jacfwd. jacfwd!
    # TODO: Replace jnp.eye with a vmap?
    jac_fn = jacfwd(f_ravel, argnums=1)  # Jacobian of f w.r.t. x
    J = jac_fn(theta, x_star_flat)

    # Solve adjoint system
    # NOTE: Do we need to transpose A? Yes!
    # NOTE: Currently not possible to cast A to a sparse matrix to use a sparse solver
    n = x_star_flat.size
    A = jnp.eye(n) - J

    b = vec.ravel()
    w = jnp.linalg.solve(A.T, b)

    # Calculate the vector Jacobian function v * df / dtheta, evaluated at at x*
    _, vjp_theta = vjp(lambda theta: f(theta, x_star), theta)

    # VJP: w * df / da
    w = w.reshape(-1, 3)
    a_bar = vjp_theta(w)

    return a_bar[0], None


# ==========================================================================
# Backward rule - Fixed point iteration
# ==========================================================================


def fixed_point_bwd_fixedpoint(
    solver: Callable,
    solver_config: dict[str, Any],
    f: Callable,
    res: tuple[SolverIterParams, Float[Array, "nodes_free 3"]],
    vec: Float[Array, "nodes_free 3"],
) -> tuple[SolverIterParams, None]:
    """
    Run the backward pass by solving the adjoint system with fixed-point iteration.

    Parameters
    ----------
    solver :
        The function that runs the fixed-point solve. Unused here.
    solver_config :
        The configuration options of the solver.
    f :
        The function iterated upon.
    res :
        The residual from the forward pass: the parameters and the fixed point.
    vec :
        The cotangent vector on the left of the VJP.

    Returns
    -------
    grads :
        The cotangent with respect to the parameters, and None for the unused
        initial guess.

    Notes
    -----
    Solves the adjoint fixed-point equation with the same forward iteration used in
    the forward pass, avoiding an explicit Jacobian.
    """
    # Unpack data from forward pass
    a, x_star = res

    # Calculate the vector Jacobian function v * df / dx at x*, closed around a
    _, vjp_x = vjp(lambda x: f(a, x), x_star)

    def rev_iter(vec, u):
        """
        Evaluates the function: u = vector + u * df / dx

        Notes
        -----
        The function ought to have signature f(a, u(a)).
        We are looking for a fixed point u*(a) = f(a, u*(a)).
        """
        return vec + vjp_x(u)[0]

    # Copy solver config to leave forward config untouched
    solver_config = {k: v for k, v in solver_config.items()}
    solver_config["eta"] = 1e-6

    # Solve adjoint function iteratively
    u_star = solver_forward(
        rev_iter,  # The function to find a fixed-point of
        vec,  # The parameters of rev_iter
        vec,  # The initial guess of the solution vector
        solver_config,  # The configuration of the solver
    )

    # Calculate the vector Jacobian function v * df / da at a, closed around x*
    _, vjp_a = vjp(lambda a: f(a, x_star), a)

    # VJP: u * df / da
    a_bar = vjp_a(u_star)

    return a_bar[0], None


# ==========================================================================
# Backward rule - Adjoint method
# ==========================================================================


def fixed_point_bwd_adjoint_general(
    solver: Callable,
    solver_config: dict[str, Any],
    f: Callable,
    res: tuple[SolverIterParams, Float[Array, "nodes_free 3"]],
    vec: Float[Array, "nodes_free 3"],
) -> tuple[SolverIterParams, None]:
    """
    Run the backward pass by solving the adjoint system with BiCGSTAB.

    Parameters
    ----------
    solver :
        The function that runs the fixed-point solve. Unused here.
    solver_config :
        The configuration options of the solver. Unused here.
    f :
        The function iterated upon.
    res :
        The residual from the forward pass: the parameters and the fixed point.
    vec :
        The cotangent vector on the left of the VJP.

    Returns
    -------
    grads :
        The cotangent with respect to the parameters, and None for the unused
        initial guess.

    Notes
    -----
    Solves the adjoint system matrix-free with the BiCGSTAB Krylov solver, trading
    the dense Jacobian of the materialized variant for iterative matrix-vector
    products.
    """
    # Unpack data from forward pass
    a, x_star = res

    # Calculate the vector Jacobian function v * df / dx at x*, closed around a
    _, vjp_x = vjp(lambda x: f(a, x), x_star)

    def A_fn(w):
        """
        Evaluates the function: w = vector + w * df / dx

        Notes
        -----
        The function ought to have signature f(a, u(a)).
        We are looking for a fixed point u*(a) = f(a, u*(a)).
        """
        return w - vjp_x(w)[0]

    # Solve adjoint function iteratively
    u_star, _ = bicgstab(A_fn, vec, tol=1e-6, atol=1e-6)

    # Calculate the vector Jacobian function v * df / da at a, closed around x*
    _, vjp_a = vjp(lambda a: f(a, x_star), a)

    # VJP: u * df / da
    a_bar = vjp_a(u_star)

    return a_bar[0], None


def fixed_point_bwd_adjoint(
    solver: Callable,
    solver_config: dict[str, Any],
    f: Callable,
    res: tuple[SolverIterParams, Float[Array, "nodes_free 3"]],
    vec: Float[Array, "nodes_free 3"],
) -> tuple[SolverIterParams, None]:
    """
    Run the backward pass with a pseudo-adjoint method reusing the stiffness matrix.

    Parameters
    ----------
    solver :
        The function that runs the fixed-point solve. Unused here.
    solver_config :
        The configuration options of the solver, read for the ``loads_fn``.
    f :
        The function iterated upon, with signature ``f(theta, x(theta))``.
    res :
        The residual from the forward pass: the parameters and the fixed point.
    vec :
        The cotangent vector on the left of the VJP.

    Returns
    -------
    grads :
        The cotangent with respect to the parameters, and None for the unused
        initial guess.

    Notes
    -----
    The adjoint operator is built from the load Jacobian and the stiffness matrix
    ``theta[0]``, then solved with conjugate gradients. When the stiffness matrix is
    sparse, its LU factorization is computed once and reused inside a custom linear
    solve. This is the registered backward rule for the implicit solver.
    """
    # Unpack data from forward pass
    theta, x_star = res

    # Get load function
    p_fn = solver_config["loads_fn"]

    # Calculate the vector Jacobian function v * dp / dx at x*, closed around theta
    _, vjp_x = vjp(lambda x: p_fn(theta, x), x_star)

    # Stiffness matrix is the first parameter
    # The matrix is symmetric, so no need to transpose it
    K = theta[0]

    # Default linear solver is dense; intentionally shadowed by the sparse
    # branch below when K is a JAXSparse
    def linearsolve_fn(b):  # pyright: ignore[reportRedeclaration]
        """
        Closure function around a dense linear solver.
        """
        return jnp.linalg.solve(K, b)

    # If the stiffness matrix is sparse, use a sparse linear solver
    if isinstance(K, JAXSparse):
        # K is the stiffness matrix theta[0]; when sparse it is always the CSC
        # built by EquilibriumModelSparse.stiffness_matrix, but isinstance only
        # narrows the opaque theta[0] to JAXSparse. K_id caches the sparse LU
        # factorization.
        K_id = splu(K)  # pyright: ignore[reportArgumentType]

        def linearsolve_fn(b):  # noqa: F811
            """
            Reuse a pre-computed LU decomposition of the stiffness matrix to solve a
            linear system.
            Also linearize it to get its transpose, which is required by lineax.
            """

            def matvec_fn(_x):
                return K @ _x

            def solve_fn(_, _b):
                return splu_solve(K_id, _b)

            return custom_linear_solve(matvec_fn, b, solve_fn, symmetric=True)

    def A_fn(w):
        """
        Evaluates the function: vector = w - w @ K_inv @ dp / dx
        """
        lam = linearsolve_fn(w)

        return w - vjp_x(lam)[0]

    # Solve adjoint function iteratively
    input_structure = jax.ShapeDtypeStruct(vec.shape, vec.dtype)
    A_op = FunctionLinearOperator(A_fn, input_structure)
    # reuses the `solver` param name for the lineax adjoint solver; unrelated to
    # the nondiff solver arg, and a lineax AbstractLinearSolver from here on
    solver = Normal(CG(rtol=1e-6, atol=1e-6))  # pyright: ignore[reportAssignmentType]
    solution = linear_solve(A_op, vec, solver, throw=False)  # pyright: ignore[reportArgumentType]
    w = solution.value

    # Calculate the vector Jacobian function v * df / dtheta, evaluated at at x*
    _, vjp_theta = vjp(lambda theta: f(theta, x_star), theta)

    # VJP: w * df / dtheta
    theta_bar = vjp_theta(w)

    return theta_bar[0], None


# ==========================================================================
# Register custom VJP
# ==========================================================================

solver_fixedpoint_implicit.defvjp(fixed_point_fwd, fixed_point_bwd_adjoint)
