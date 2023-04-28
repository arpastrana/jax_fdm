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

def spsolve_gpu2(data, indices, indptr, b):
    """
    A wrapper around cuda sparse linear solver that is GPU friendly.
    """
    # TODO: probably needs transformation of data, indices and indptr from CSC to CSR format.
    # TODO: what would happen if we pass CSC data, indices and indptr to CUDA sparse solve?
    # TODO: is csrlsvqr the CUDA sparse solver we actually need? what about csrlsvchol?
    # csr = csc_matrix((data, indices, indptr)).tocsr()

    # NOTE: we can pass csc indices directly because we can!
    # Just kidding. This is because the matrix A is symmetric :)
    return spsolve_jax(data, indices, indptr, b[:, 0])


def spsolve_gpu(data, indices, indptr, b):
    """
    A wrapper around cuda sparse linear solver that is GPU friendly.
    """
    x = spsolve_jax(data, indices, indptr, b[:, 0])
    y = spsolve_jax(data, indices, indptr, b[:, 1])
    z = spsolve_jax(data, indices, indptr, b[:, 2])

    return jnp.concatenate((x, y, z))


def spsolve_cpu(data, indices, indptr, b):
    """
    A wrapper around scipy sparse linear solver.
    """
    csc = csc_matrix((data, indices, indptr))

    return spsolve_scipy(csc, b)


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

sparse_solver = register_sparse_solver(solvers)


# ==========================================================================
# Define sparse linear solver
# ==========================================================================

def linear_solve_callback(data, indices, indptr, b):
    """
    The linear solve callback.
    """
    return sparse_solver(data, indices, indptr, b)


@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def linear_solve(q, xyz_fixed, loads, free, c_free, c_fixed, index_array, diag_indices, diags):
    """
    The sparse linear solver.
    """
    A = force_densities_to_A(q, index_array, diag_indices, diags)
    b = loads[free, :] - c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))

    # NOTE: GPU sparse solver does not need pure callback
    xk = jax.pure_callback(linear_solve_callback,
                           b,  # return type is b
                           A.data, A.indices, A.indptr, b)  # input arguments

    xk = spsolve_gpu(...)  # GPU variant

    return xk


# ==========================================================================
# Forward and backward passes
# ==========================================================================

def linear_solve_fwd(q, xyz_fixed, loads, free, c_free, c_fixed, index_array, diag_indices, diags):
    """
    Forward pass of the sparse linear solver.

    Call the linear solve and save parameters and solution for the backward pass.
    """
    xk = linear_solve(q, xyz_fixed, loads, free, c_free, c_fixed, index_array, diag_indices, diags)

    # lhs_matrix = force_densities_to_A(q, index_array, diag_indices, diags)

    return xk, (xk, q, xyz_fixed, loads)


def linear_solve_bwd(free, c_free, c_fixed, index_array, diag_indices, diags, res, g):
    """
    Backward pass of the sparse linear solver.
    """
    xk, q, xyz_fixed, loads = res

    # function that translates parameters into LHS matrix in CSC format
    A = force_densities_to_A(q, index_array, diag_indices, diags)

    # Solve adjoint system
    # A.T @ xk_bar = -g
    lam = jax.pure_callback(linear_solve_callback,
                            g,  # return type is g
                            A.data, A.indices, A.indptr, g)  # input arguments

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


def get_sparse_diag_indices(csc):
    """
    Given a CSC matrix, get indices into `data` that access diagonal elements in order.
    """
    all_indices = []
    for i in range(csc.shape[1]):
        index_range = csc.indices[csc.indptr[i]:csc.indptr[i + 1]]
        ind_loc = jnp.where(index_range == i)[0]
        all_indices.append(ind_loc + csc.indptr[i])

    return jnp.concatenate(all_indices)


# ==========================================================================
# Register forward and backward passes to JAX
# ==========================================================================

linear_solve.defvjp(linear_solve_fwd, linear_solve_bwd)
