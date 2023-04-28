"""
NOTE: Sparse solver does not support forward mode auto-differentiation yet.
"""
from functools import partial

import jax
import jax.numpy as jnp

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as spsolve_scipy

from jax.experimental.sparse import CSC
from jax.experimental.sparse.linalg import spsolve as spsolve_jax


# ==========================================================================
# Register sparse linear solvers
# ==========================================================================

def spsolve_gpu(data, indices, indptr, b):
    """
    A wrapper around cuda sparse linear solver that is GPU friendly.

    Notes
    -----
    We must split b into three vectors (one per spatial coordinate),
    and then solve a sparse system 3 times because the CUDA sparse solver
    in the backend cannot take b as a matrix.

    The sparse solve in a GPU is thus 3x more expensive than if it could
    solve for the b matrix all at once.

    This limitation with CUDA might make a GPU sparse solve more expensive
    than a CPU sparse solve. So use this method with a pinch of salt.
    """
    # NOTE: we can pass csc indices directly because we can!
    # Just kidding. This is because the matrix A is symmetric :)
    # TODO: Ravel and unravel this!
    x = spsolve_jax(data, indices, indptr, b[:, 0])
    y = spsolve_jax(data, indices, indptr, b[:, 1])
    z = spsolve_jax(data, indices, indptr, b[:, 2])

    return jnp.concatenate((x, y, z))


def spsolve_cpu(A, b):
    """
    A wrapper around scipy sparse linear solver that acts as a JAX pure callback.
    """
    def callback(data, indices, indptr, _b):
        _A = csc_matrix((data, indices, indptr))
        return spsolve_scipy(_A, _b)

    xk = jax.pure_callback(callback,  # callback function
                           b,  # return type is b
                           A.data,  # callback function arguments from here on
                           A.indices,
                           A.indptr,
                           b)

    return xk


def register_sparse_solver(solvers):
    """
    Register the sparse solver used by the FDM model based on JAX default backend.
    """
    backend = jax.default_backend()
    sparse_solver = solvers.get(backend)

    if not sparse_solver:
        raise ValueError(f"Default backend {backend} does not support a sparse solver")

    return sparse_solver


solvers = {"cpu": spsolve_cpu,
           "gpu": spsolve_gpu}

spsolve = register_sparse_solver(solvers)


# ==========================================================================
# Define sparse linear solver
# ==========================================================================

@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def sparse_solve(q, xyz_fixed, loads, free, c_free, c_fixed, index_array, diag_indices, diags):
    """
    The sparse linear solver.
    """
    A = force_densities_to_A(q, index_array, diag_indices, diags)
    b = loads[free, :] - c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))

    return spsolve(A, b)


# ==========================================================================
# Forward and backward passes
# ==========================================================================

def sparse_solve_fwd(q, xyz_fixed, loads, free, c_free, c_fixed, index_array, diag_indices, diags):
    """
    Forward pass of the sparse linear solver.

    Call the linear solve and save parameters and solution for the backward pass.
    """
    xk = sparse_solve(q, xyz_fixed, loads, free, c_free, c_fixed, index_array, diag_indices, diags)

    return xk, (xk, q, xyz_fixed, loads)


def sparse_solve_bwd(free, c_free, c_fixed, index_array, diag_indices, diags, res, g):
    """
    Backward pass of the sparse linear solver.
    """
    xk, q, xyz_fixed, loads = res

    # function that translates parameters into LHS matrix in CSC format
    A = force_densities_to_A(q, index_array, diag_indices, diags)

    # Solve adjoint system
    # A.T @ xk_bar = -g
    lam = spsolve(A, g)

    # the implicit constraint function for implicit differentiation
    def residual_fn(params):
        q, xyz_fixed, loads = params
        b = loads[free, :] - c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))
        A = force_densities_to_A(q, index_array, diag_indices, diags)
        return b - A @ xk

    params = (q, xyz_fixed, loads)

    # Call vjp of residual_fn to compute gradient wrt params
    params_bar = jax.vjp(residual_fn, params)[1](lam)[0]

    return params_bar


# ==========================================================================
# Helpers
# ==========================================================================

def force_densities_to_A(q, index_array, diag_indices, diags):
    """
    Computes the LHS matrix in CSC format from a given vector of force densities.
    """
    nondiags_data = -q[index_array.data - 1]
    nondiags = CSC((nondiags_data, index_array.indices, index_array.indptr), shape=index_array.shape)
    diag_fd = diags.T @ q  # sum of force densities for each node

    nondiags.data = nondiags.data.at[diag_indices].set(diag_fd)

    return nondiags


# ==========================================================================
# Register forward and backward passes to JAX
# ==========================================================================

sparse_solve.defvjp(sparse_solve_fwd, sparse_solve_bwd)
