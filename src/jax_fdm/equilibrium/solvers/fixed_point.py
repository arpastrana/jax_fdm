from functools import partial

import jax

import matplotlib.pyplot as plt

from jaxopt.linear_solve import solve_cg
from jaxopt.linear_solve import solve_normal_cg

from jax.scipy.sparse.linalg import gmres
from jax.scipy.sparse.linalg import cg

from jax.scipy.linalg import block_diag

import lineax as lx

import jax.numpy as jnp

from jax import custom_vjp
from jax import jacrev
from jax import jacfwd
from jax import vjp
from jax import jvp

from equinox.internal import while_loop

from jaxopt import FixedPointIteration
from jaxopt import AndersonAcceleration

from jax_fdm.equilibrium.solvers.jaxopt import solver_jaxopt


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
        solver_newton
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
# Fixed point solver wrapper for implicit differentiation
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


# ==========================================================================
# Backward rules
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


def fixed_point_bwd_iterative(solver, solver_config, f, res, vec):
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
    # u_star, info = gmres(A_fn, vec, x0=vec, tol=1e-6)
    u_star, info = cg(A_fn, vec, x0=vec, tol=1e-6)

    # Calculate the vector Jacobian function v * df / da at a, closed around x*
    _, vjp_a = vjp(lambda a: f(a, x_star), a)

    # VJP: u * df / da
    a_bar = vjp_a(u_star)

    return a_bar[0], None


def fixed_point_bwd_direct(solver, solver_config, f, res, vec):
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

    print(f"{J.shape=}")
    plt.spy(J, precision=1e-4)
    plt.show()
    raise

    b = vec.ravel()
    w = jnp.linalg.solve(A.T, b)

    # Calculate the vector Jacobian function v * df / dtheta, evaluated at at x*
    _, vjp_theta = vjp(lambda theta: f(theta, x_star), theta)

    # VJP: w * df / da
    w = w.reshape(-1, 3)
    a_bar = vjp_theta(w)

    return a_bar[0], None


# ==========================================================================
# Backward rule - Adjoint method
# ==========================================================================

def fixed_point_bwd_adjoint_direct(solver, solver_config, f, res, vec):
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
    print("\n*** Using direct adjoint ***\n")
    # Unpack data from forward pass
    theta, x_star = res

    # Stiffness matrix is the first parameter
    K = theta[0]

    # Get load function
    p_fn = solver_config["loads_fn"]

    # Format data
    x_star_flat = x_star.ravel()

    def p_fn_ravel(x):
        x = x.reshape(-1, 3)
        return p_fn(theta, x).ravel()

    # NOTE: Use jacrev or jacfwd. jacfwd for speed!
    jac_fn = jacfwd(p_fn_ravel)  # Jacobian of p w.r.t. x
    J = jac_fn(x_star_flat)

    K_block = block_diag(K, K, K)

    # Solve adjoint system
    # We need to transpose the jacobian to account for the LHS reordering
    A = K_block - J
    b = K_block @ vec.ravel()
    w = jnp.linalg.solve(A, b)

    # Calculate the vector Jacobian function v * df / dtheta, evaluated at at x*
    _, vjp_theta = vjp(lambda theta: f(theta, x_star), theta)

    # VJP: w * df / da
    w = w.reshape(-1, 3)
    a_bar = vjp_theta(w)

    return a_bar[0], None


def fixed_point_bwd_adjoint_iterative(solver, solver_config, f, res, vec):
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

    # Stiffness matrix is the first parameter
    K = theta[0]

    # Get load function
    loads_fn = solver_config["loads_fn"]

    # Calculate the vector Jacobian function v * df / dx at x*, closed around a
    # TODO: Do we transpose the output of loads_fn?
    _, vjp_x = vjp(lambda x: loads_fn(theta, x), x_star)

    # LHS function: (K^T − G^T) @ w = K^T @ v.
    # We don't need to transpose K because it is symmetric
    # TODO: Use jvp instead of vjp?
    def A_fn(w):
        return K @ w - vjp_x(w)[0]

    # def A_fn(w):
    #     primals, tangents = jvp(lambda x: loads_fn(theta, x), (x_star, ), (w, ))
    #     return K @ w - tangents[0].T

    # RHS matrix
    b = K @ vec

    # Solve adjoint function iteratively
    # w, info = gmres(A_fn, b, x0=vec, tol=1e-6)
    w, info = cg(A_fn, b, tol=1e-6, atol=1e-6, maxiter=10000)

    # Calculate the vector Jacobian function v * df / dtheta, evaluated at at x*
    _, vjp_theta = vjp(lambda theta: f(theta, x_star), theta)

    # VJP: w * df / da
    a_bar = vjp_theta(w)

    return a_bar[0], None


def fixed_point_bwd_adjoint_test_a(solver, solver_config, f, res, vec):
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
    print("\n*** Using test adjoint A ***\n")
    # Unpack data from forward pass
    theta, x_star = res

    # Stiffness matrix is the first parameter
    K = theta[0]

    # Get load function
    p_fn = solver_config["loads_fn"]

    # Format data
    x_star_flat = x_star.ravel()

    def p_fn_ravel(x):
        x = x.reshape(-1, 3)
        return p_fn(theta, x).ravel()

    # NOTE: Use jacrev or jacfwd. jacfwd for speed!
    jac_fn = jacfwd(p_fn_ravel)  # Jacobian of p w.r.t. x
    J = jac_fn(x_star_flat)

    # Solve adjoint system
    # We need to transpose the jacobian to account for the LHS reordering
    K_block = block_diag(K, K, K)
    G = jnp.linalg.solve(K_block, J)

    print(f"{G.shape=}")
    plt.spy(G, precision=1e-4)
    plt.show()
    raise

    n = x_star_flat.size
    A = jnp.eye(n) - G

    b = vec.ravel()
    w = jnp.linalg.solve(A.T, b)

    # Calculate the vector Jacobian function v * df / dtheta, evaluated at at x*
    _, vjp_theta = vjp(lambda theta: f(theta, x_star), theta)

    # VJP: w * df / da
    w = w.reshape(-1, 3)
    a_bar = vjp_theta(w)

    return a_bar[0], None


def fixed_point_bwd_adjoint_test(solver, solver_config, f, res, vec):
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
    print("\n*** Using test adjoint lineax ***\n")
    # Unpack data from forward pass
    theta, x_star = res

    # Stiffness matrix is the first parameter
    # The matrix is symmetric, so no need to transpose it
    K = theta[0]

    # Get load function
    p_fn = solver_config["loads_fn"]

    # Calculate the vector Jacobian function v * dp / dx at x*, closed around theta
    _, vjp_x = vjp(lambda x: p_fn(theta, x), x_star)

    # Get linear solver for stiffness matrix from equilibrium model
    # It is jax.numpy.linalg for dense matrices and scipy.spsolve for sparse matrices.
    linearsolve_fn = solver_config["linearsolve_fn"]

    def A_fn(w):
        """
        Evaluates the function: w = vector - w @ K_inv @ dp / dx
        """
        lam = linearsolve_fn(K, w)

        return w - vjp_x(lam)[0]

    # Solve adjoint function iteratively
    # w, info = cg(A_fn, vec, maxiter=1000)
    # w = solve_normal_cg(A_fn, vec)
    #
    A_op = lx.FunctionLinearOperator(A_fn, input_structure=jax.ShapeDtypeStruct(vec.shape, vec.dtype))
    lin_solver = lx.NormalCG(rtol=1e-6, atol=1e-6)
    sol = lx.linear_solve(A_op, vec, lin_solver, throw=False, options={"y0": vec})

    w = sol.value

    # Calculate the vector Jacobian function v * df / dtheta, evaluated at at x*
    _, vjp_theta = vjp(lambda theta: f(theta, x_star), theta)

    # VJP: w * df / da
    a_bar = vjp_theta(w)

    return a_bar[0], None


# ==========================================================================
# Register custom VJP
# ==========================================================================

# fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd_fixedpoint)
# fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd_iterative)
# fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd_direct)
fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd_adjoint_test)
# fixed_point.defvjp(fixed_point_fwd, fixed_point_bwd_adjoint_iterative)
