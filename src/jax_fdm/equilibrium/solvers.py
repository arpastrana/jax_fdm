from functools import partial

import jax

import jax.numpy as jnp

from jax import vjp
from jax import custom_vjp
from jax import jacrev

from equinox.internal import while_loop

from jaxopt import FixedPointIteration
from jaxopt import AndersonAcceleration


# ==========================================================================
# Iterative solvers
# ==========================================================================

def solver_anderson(f, a, x_init, solver_config):
    """
    Solve for a fixed point of a function f(a, x) using anderson acceleration in jaxopt.

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
    implicit_diff = solver_config["implicit"]

    def f_swapped(x, a):
        return f(a, x)

    fpi = AndersonAcceleration(fixed_point_fun=f_swapped,
                               maxiter=tmax,
                               tol=eta,  # 1e-5 is the jaxopt default
                               has_aux=False,
                               history_size=5,  # 5 is the jaxopt default
                               ridge=1e-6,  # 1e-5 is the jaxopt default
                               implicit_diff=implicit_diff,
                               verbose=verbose)

    result = fpi.run(x_init, a)

    return result.params


def solver_fixedpoint(f, a, x_init, solver_config):
    """
    Solve for a fixed point of a function f(a, x) using forward iteration in jaxopt.

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

    def f_swapped(x, a):
        return f(a, x)

    fpi = FixedPointIteration(fixed_point_fun=f_swapped,
                              maxiter=tmax,
                              tol=eta,
                              has_aux=False,
                              # implicit_diff=True,
                              # jit=True,
                              # unroll=False,
                              verbose=verbose)

    result = fpi.run(x_init, a)

    return result.params


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


def solver_newton(f, a, x_init, solver_config):
    """
    Find a root of the equation f(a, x) - x = 0 using Newton's method.

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
    def f_root(x):
        return f(a, x) - x

    u, v = x_init.shape
    nrows = ncols = u * v

    def f_newton(a, x):
        b = f_root(x)
        jinv = jacrev(f_root)(x)
        jinv = jnp.reshape(jinv, (nrows, ncols))
        b = jnp.reshape(b, (-1, 1))
        step = jnp.linalg.solve(jinv, b)
        step = jnp.reshape(step, (-1, 3))

        return x - step

    x_star = solver_forward(f_newton, a, x_init, solver_config)

    return jnp.reshape(x_star, (-1, 3))


# ==========================================================================
# Fixed point solver function
# ==========================================================================

@partial(custom_vjp, nondiff_argnums=(0, 1, 2))
def fixed_point(solver, solver_config, f, a, x_init):
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
    x_star = fixed_point(solver, solver_config, f, a, x_init)

    return x_star, (a, x_star)


def fixed_point_bwd(solver, solver_config, f, res, vec):
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
    a, x_star = res
    _, vjp_a = vjp(lambda a: f(a, x_star), a)

    def rev_iter(packed, u):
        """
        The function ought to have signature f(a, u(a)).
        We are looking for a fixed point u*(a) = f(a, u*(a)).
        """
        a, x_star, vec = packed

        # Calculate the Jacobian df / dx
        _, vjp_x = vjp(lambda x: f(a, x), x_star)

        # Affine function: u = vector + u * df / dx
        return vec + vjp_x(u)[0]

    u_star = solver(
        rev_iter,  # The function to find a fixed-point of
        (a, x_star, vec),  # The parameters of rev_iter
        vec,  # The initial guess of the solution vector
        solver_config)  # The configuration of the solver

    # VJP: u * df / da
    a_bar = vjp_a(u_star)[0]

    return a_bar, None


fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd)
