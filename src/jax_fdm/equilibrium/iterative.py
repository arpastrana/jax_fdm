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
    """
    tmax = solver_config["tmax"]
    eta = solver_config["eta"]
    verbose = solver_config["verbose"]

    def f_swapped(x, a):
        return f(a, x)

    fpi = AndersonAcceleration(fixed_point_fun=f_swapped,
                               maxiter=tmax,
                               tol=eta,  # 1e-5 is the default,
                               has_aux=False,
                               history_size=5,  # 5 is default
                               ridge=1e-5,  # 1e-5 is the default
                               # implicit_diff=True,
                               # jit=True,
                               # unroll=False,
                               verbose=verbose)

    result = fpi.run(x_init, a)

    return result.params


def solver_fixedpoint(f, a, x_init, solver_config):
    """
    Solve for a fixed point of a function f(a, x) using forward iteration in jaxopt.
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
    Solve for a fixed point of a function f(a, x) using forward iteration.
    """
    return solver(f, a, x_init, solver_config)


def fixed_point_fwd(solver, solver_config, f, a, x_init):
    """
    The forward pass of a fixed point solver.
    """
    x_star = fixed_point(solver, solver_config, f, a, x_init)
    return x_star, (a, x_star)


def fixed_point_bwd(solver, solver_config, fn, res, x_star_bar):
    """
    The backward pass of a fixed point solver.
    """
    a, x_star = res
    _, vjp_a = vjp(lambda a: fn(a, x_star), a)

    def rev_iter(packed, u):
        a, x_star, x_star_bar = packed
        _, vjp_x = vjp(lambda x: fn(a, x), x_star)
        return x_star_bar + vjp_x(u)[0]

    # solver_config = {k: v for k, v in solver_config.items()}
    # solver_config["eta"] = 1e-3
    partial_func = solver(rev_iter,
                          (a, x_star, x_star_bar),
                          x_star_bar,
                          solver_config)

    a_bar = vjp_a(partial_func)[0]

    return a_bar, None


fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd)
