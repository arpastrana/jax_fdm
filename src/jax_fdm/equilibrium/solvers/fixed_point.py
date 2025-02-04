from functools import partial

import jax

from jax.scipy.sparse.linalg import gmres
from jax.scipy.sparse.linalg import cg

import jax.numpy as jnp

from jax import custom_vjp
from jax import jacrev
from jax import jacfwd
from jax import vjp

from equinox.internal import while_loop

from jaxopt import FixedPointIteration
from jaxopt import AndersonAcceleration

from jax_fdm.equilibrium.solvers.jaxopt import solver_jaxopt


# ==========================================================================
# Iterative solvers - JAXOPT
# ==========================================================================

def solver_anderson(f, solver_config):
    """
    Find a fixed point of a function f(x, theta) using Anderson acceleration.

    Parameters
    ----------
    f : The function to iterate upon.
    solver_config: The configuration options of the solver.

    Returns
    -------
    x_star : The solution vector at the fixed point.
    """
    solver_kwargs = {"history_size": 5, "ridge": 1e-6}

    return solver_jaxopt(AndersonAcceleration, f, solver_config, solver_kwargs)


def solver_fixedpoint(f, solver_config):
    """
    Find a fixed point of a function f(x, theta) using fixed point iteration.

    Parameters
    ----------
    f : The function to iterate upon.
    solver_config: The configuration options of the solver.

    Returns
    -------
    x_star : The solution vector at the fixed point.
    """
    return solver_jaxopt(FixedPointIteration, f, solver_config)


def is_solver_fixedpoint(solver_fn):
    """
    Test if a solver function is a fixed point solver function.

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
        solver_newton
    }

    return solver_fn in solver_fns


# ==========================================================================
# Homecooked solvers
# ==========================================================================

def solver_forward(f, x_init, theta, solver_config):
    """
    Solve for a fixed point of a function f(x, theta) using forward iteration.

    Parameters
    ----------
    f : The function to iterate upon.
    x_init: An initial guess for the values of the solution vector.
    theta : The function parameters.
    solver_config: The configuration options of the solver.

    Returns
    -------
    x_star : The solution vector at a fixed point.
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
        return x, f(x, theta)

    init_val = (x_init, f(x_init, theta))

    _, x_star = while_loop(cond_fun,
                           body_fun,
                           init_val,
                           max_steps=tmax,
                           kind="checkpointed")

    return x_star


def solver_newton(f, x_init, theta, solver_config):
    """
    Find a root of the equation f(x, theta) - x = 0 using Newton's method.

    Parameters
    ----------
    f : The function to iterate upon.
    x_init: An initial guess for the values of the solution vector.
    theta : The function parameters.
    solver_config: The configuration options of the solver.

    Returns
    -------
    x_star : The solution vector at a fixed point.
    """
    def f_root(x):
        return f(x, theta) - x

    u, v = x_init.shape
    nrows = ncols = u * v

    def f_newton(x, theta):
        b = f_root(x)
        jinv = jacrev(f_root)(x)
        jinv = jnp.reshape(jinv, (nrows, ncols))
        b = jnp.reshape(b, (-1, 1))
        step = jnp.linalg.solve(jinv, b)
        step = jnp.reshape(step, (-1, 3))

        return x - step

    x_star = solver_forward(f_newton, x_init, theta, solver_config)

    return jnp.reshape(x_star, (-1, 3))


# ==========================================================================
# Fixed point solver wrapper for implicit differentiation
# ==========================================================================

@partial(custom_vjp, nondiff_argnums=(0, ))
def fixed_point(solver, x_init, theta):
    """
    Solve for a fixed point of a function f(x, theta) using an iterative solver.
    """
    return solver(x_init, theta)


def fixed_point_fwd(solver, x_init, theta):
    """
    The forward pass of an iterative fixed point solver.

    Parameters
    ----------
    solver: The function that executes a fixed point solver.
    solver_config: The configuration options of the solver.
    f : The function f(x, theta) to iterate upon.
    theta : The function parameters.
    x_init: An initial guess for the values of the solution vector.

    Returns
    -------
    x : The solution vector at a fixed point.
    res : Auxiliary data to transfer to the backward pass.
    """
    x_star = fixed_point(solver, x_init, theta)

    return x_star, (x_star, theta)


def fixed_point_bwd_forward(solver, res, vec):
    """
    The backward pass of an iterative fixed point solver.

    Parameters
    ----------
    solver: The function that executes a fixed point solver.
    solver_config: The configuration options of the solver.
    f : The function f(x, theta) to iterate upon.
    res : Auxiliary data transferred from the forward pass.
    vec: The vector on the left of the VJP.

    Returns
    -------
    x : The solution vector at a fixed point.
    res : None
    """
    # Fetch fixed point function from solver
    f = solver.fixed_point_fun

    # Unpack data from forward pass
    x_star, theta = res

    # Calculate the vector Jacobian function v * df / dx at x*, closed around theta
    _, vjp_x = vjp(lambda x: f(x, theta), x_star)

    def rev_iter(vec, u):
        """
        Evaluates the function: u = vector + u * df / dx

        Notes
        -----
        The function ought to have signature f(theta, u(theta)).
        We are looking for a fixed point u*(theta) = f(theta, u*(theta)).
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

    # Calculate the vector Jacobian function v * df / dtheta at theta, closed around x*
    _, vjp_theta = vjp(lambda theta: f(x_star, theta), theta)

    # VJP: u * df / dtheta
    theta_bar = vjp_theta(u_star)

    return theta_bar[0], None


def fixed_point_bwd_iterative(solver, res, vec):
    """
    The backward pass of an iterative fixed point solver.

    Parameters
    ----------
    solver: The function that executes a fixed point solver.
    solver_config: The configuration options of the solver.
    f : The function f(x, theta) to iterate upon.
    res : Auxiliary data transferred from the forward pass.
    vec: The vector on the left of the VJP.

    Returns
    -------
    x : The solution vector at a fixed point.
    res : None
    """
    # Fetch fixed point function from solver
    f = solver.fixed_point_fun

    # Unpack data from forward pass
    x_star, theta = res

    # Calculate the vector Jacobian function v * df / dx at x*, closed around theta
    _, vjp_x = vjp(lambda x: f(x, theta), x_star)

    def A_fn(w):
        """
        Evaluates the function: w = vector + w * df / dx

        Notes
        -----
        The function ought to have signature f(theta, u(theta)).
        We are looking for a fixed point u*(theta) = f(theta, u*(theta)).
        """
        return w - vjp_x(w)[0]

    # Solve adjoint function iteratively
    # u_star, info = gmres(A_fn, vec, x0=vec, tol=1e-6)
    u_star, info = cg(A_fn, vec, x0=vec, tol=1e-6)

    # Calculate the vector Jacobian function v * df / dtheta at theta, closed around x*
    _, vjp_theta = vjp(lambda theta: f(x_star, theta), theta)

    # VJP: u * df / dtheta
    theta_bar = vjp_theta(u_star)

    return theta_bar[0], None


def fixed_point_bwd_direct(solver, res, vec):
    """
    The backward pass of an iterative fixed point solver.

    Parameters
    ----------
    solver: The function that executes a fixed point solver.
    solver_config: The configuration options of the solver.
    f : The function f(x, theta) to iterate upon.
    res : Auxiliary data transferred from the forward pass.
    vec: The vector on the left of the VJP.

    Returns
    -------
    x : The solution vector at a fixed point.
    res : None
    """
    # Fetch fixed point function from solver
    f = solver.fixed_point_fun

    # Unpack data from forward pass
    x_star, theta = res

    # Format data
    x_star_flat = x_star.ravel()
    n = x_star_flat.size

    def f_ravel(x, theta):
        x = x.reshape(-1, 3)
        return f(x, theta).ravel()

    # NOTE: Use jacrev or jacfwd. jacfwd!
    # TODO: Replace jnp.eye with a vmap?
    jac_fn = jacfwd(f_ravel, argnums=0)  # Jacobian of f w.r.t. x
    J = jac_fn(x_star_flat, theta)

    # Solve adjoint system
    # NOTE: Do we need to transpose A? Yes!
    # NOTE: Currently not possible to cast A to a sparse matrix to use a sparse solver
    A = jnp.eye(n) - J
    b = vec.ravel()
    w = jnp.linalg.solve(A.T, b)

    # Calculate the vector Jacobian function v * df / dtheta, evaluated at at x*
    _, vjp_theta = vjp(lambda theta: f(x_star, theta), theta)

    # VJP: w * df / dtheta
    w = w.reshape(-1, 3)
    theta_bar = vjp_theta(w)

    return theta_bar[0], None


# ==========================================================================
# Register custom VJP
# ==========================================================================

# fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd_forward)
# fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd_iterative)
fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd_direct)
