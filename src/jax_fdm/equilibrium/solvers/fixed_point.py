from functools import partial

import jax

from jax.scipy.sparse.linalg import bicgstab
from jax.experimental.sparse import JAXSparse

import jax.numpy as jnp

from jax import custom_vjp
from jax import jacfwd
from jax import vjp

from jax.lax import custom_linear_solve

from jaxopt import FixedPointIteration
from jaxopt import AndersonAcceleration

from equinox.internal import while_loop

from lineax import FunctionLinearOperator
from lineax import NormalCG
from lineax import linear_solve

from jax_fdm.equilibrium.solvers.jaxopt import solver_jaxopt
from jax_fdm.equilibrium.sparse import splu_cpu as splu
from jax_fdm.equilibrium.sparse import splu_solve_cpu as splu_solve


# ==========================================================================
# Iterative solvers - JAXOPT
# ==========================================================================

def solver_anderson(f, a, x_init, solver_config):
    """
    Find a fixed point of a function f(a, x) using Anderson acceleration.

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
    solver_kwargs = {"history_size": 5, "ridge": 1e-6}

    return solver_jaxopt(AndersonAcceleration, f, a, x_init, solver_config, solver_kwargs)


def solver_fixedpoint(f, a, x_init, solver_config):
    """
    Find a fixed point of a function f(a, x) using Anderson acceleration.

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
    return solver_jaxopt(FixedPointIteration, f, a, x_init, solver_config)


def is_solver_fixedpoint(solver_fn):
    """
    Test if a solver function is a fixed point solver.

    Parameters
    ----------
    `solver_fn`: A solver function

    Returns
    -------
    `True` if the solver is a fixed point solver. Otherwise, `False`.
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

def solver_forward(f, a, x_init, solver_config):
    """
    Solve for a fixed point of a function f(a, x) using forward iteration.

    Parameters
    ----------
    f : The function to iterate upon.
    a : The function parameters.
    x_init: An initial guess for the values of the solution vector.
    solver_config: The configuration options of the solver.

    Returns
    -------
    x : The solution vector at a fixed point.
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

    _, x_star = while_loop(cond_fun,
                           body_fun,
                           init_val,
                           max_steps=tmax,
                           kind="checkpointed")

    return x_star


# ==========================================================================
# Fixed point solver wrapper for implicit differentiation
# ==========================================================================

@partial(custom_vjp, nondiff_argnums=(0, 1, 2))
def solver_fixedpoint_implicit(solver, solver_config, f, a, x_init):
    """
    Solve for a fixed point of a function f(a, x) using an iterative solver.
    """
    return solver(f, a, x_init, solver_config)


def fixed_point_fwd(solver, solver_config, f, a, x_init):
    """
    The forward pass of an iterative fixed point solver.

    Parameters
    ----------
    solver: The function that executes a fixed point solver.
    solver_config: The configuration options of the solver.
    fn : The function to iterate upon.
    a : The function parameters.
    x_init: An initial guess for the values of the solution vector.

    Returns
    -------
    x : The solution vector at a fixed point.
    res : Auxiliary data to transfer to the backward pass.
    """
    x_star = solver_fixedpoint_implicit(solver, solver_config, f, a, x_init)

    return x_star, (a, x_star)


# ==========================================================================
# Backward rule - Materialize Jacobian
# ==========================================================================

def fixed_point_bwd_materialize(solver, solver_config, f, res, vec):
    """
    The backward pass of a fixed point solver materializing the Jacobian.

    Parameters
    ----------
    solver: The function that executes a fixed point solver.
    solver_config: The configuration options of the solver.
    f : The function to iterate upon. It ought to have signature f(theta, x(theta)).
    res : Auxiliary data transferred from the forward pass.
    vec: The vector on the left of the VJP.

    Returns
    -------
    x : The solution vector at a fixed point.
    res : None
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

def fixed_point_bwd_fixedpoint(solver, solver_config, f, res, vec):
    """
    The backward pass of an iterative fixed point solver.

    Parameters
    ----------
    solver: The function that executes a fixed point solver.
    solver_config: The configuration options of the solver.
    f : The function to iterate upon.
    res : Auxiliary data transferred from the forward pass.
    vec: The vector on the left of the VJP.

    Returns
    -------
    x : The solution vector at a fixed point.
    res : None
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
        solver_config  # The configuration of the solver
    )

    # Calculate the vector Jacobian function v * df / da at a, closed around x*
    _, vjp_a = vjp(lambda a: f(a, x_star), a)

    # VJP: u * df / da
    a_bar = vjp_a(u_star)

    return a_bar[0], None


# ==========================================================================
# Backward rule - Adjoint method
# ==========================================================================

def fixed_point_bwd_adjoint_general(solver, solver_config, f, res, vec):
    """
    The backward pass of a fixed point solver with the adjoint method.

    Parameters
    ----------
    solver: The function that executes a fixed point solver.
    solver_config: The configuration options of the solver.
    f : The function to iterate upon.
    res : Auxiliary data transferred from the forward pass.
    vec: The vector on the left of the VJP.

    Returns
    -------
    x : The solution vector at a fixed point.
    res : None
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


def fixed_point_bwd_adjoint(solver, solver_config, f, res, vec):
    """
    The backward pass of an iterative fixed point solver with a pseudo-adjoint method.

    Parameters
    ----------
    solver: The function that executes a fixed point solver.
    solver_config: The configuration options of the solver.
    f : The function to iterate upon. It ought to have signature f(theta, x(theta)).
    res : Auxiliary data transferred from the forward pass.
    vec: The vector on the left of the VJP.

    Returns
    -------
    x : The solution vector at a fixed point.
    res : None
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

    # Default linear solver is dense
    def linearsolve_fn(b):
        """
        Closure function around a dense linear solver.
        """
        return jnp.linalg.solve(K, b)

    # If the stiffness matrix is sparse, use a sparse linear solver
    if isinstance(K, JAXSparse):
        K_id = splu(K)  # Session ID of the cached sparse LU factorization

        def linearsolve_fn(b):
            """
            Reuse a pre-computed LU decomposition of the stiffness matrix to solve a linear system.
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
    solver = NormalCG(rtol=1e-6, atol=1e-6)
    solution = linear_solve(A_op, vec, solver, throw=False)
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
