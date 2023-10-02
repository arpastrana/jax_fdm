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
# Sparse linear solver on GPU
# ==========================================================================

def spsolve_gpu_ravel(A, b):
    """
    A wrapper around cuda sparse linear solver that is GPU friendly.

    Notes
    -----
    We ravel b into a single vector because the CUDA sparse solver backend
    cannot take b as a matrix.
    Accordingly, we must construct a sparse block diagonal matrix that
    repeats matrix A three times (one time per spatial coordinate XYZ).

    This "raveled" linear solver could be faster than the `spsolve_gpu_stack`
    alternative, because this functions solves the block linear system
    all at once instead of 3 times in sequence.

    This limitation with CUDA might make a GPU sparse solve more expensive
    than a CPU sparse solve. So use this method with a pinch of salt.
    """
    # NOTE: we can pass CSC indices directly to the JAX solver because we can!
    # Just kidding. This is because the matrix A is symmetric :)
    A = sparse_blockdiag_matrix(A, 3)
    b = jnp.ravel(b, "F")
    X = spsolve_jax(A.data, A.indices, A.indptr, b)

    return jnp.reshape(X, (3, -1)).T


def spsolve_gpu_stack(A, b):
    """
    A wrapper around cuda sparse linear solver that is GPU friendly.

    Notes
    -----
    We must split b into three vectors (one per spatial coordinate XYZ),
    and then solve a sparse system 3 times because the CUDA sparse solver
    in the backend cannot take b as a matrix, but only as a vector.

    The sparse solve in a GPU could thus be 3x more expensive than if it could
    solve for the b matrix all at once?

    This limitation with CUDA might make a GPU sparse solve more expensive
    than a CPU sparse solve. So use this method with a pinch of salt.
    """
    # NOTE: JAX requires a csr matrix as input,
    # but we can pass csc indices directly because we can!
    # Just kidding. This is because the matrix A is symmetric :)

    # TODO: Ravel and unravel this!
    x = spsolve_jax(A.data, A.indices, A.indptr, b[:, 0])
    y = spsolve_jax(A.data, A.indices, A.indptr, b[:, 1])
    z = spsolve_jax(A.data, A.indices, A.indptr, b[:, 2])

    return jnp.stack((x, y, z), axis=1)


# Set ravel solver as the GPU sparse solver
spsolve_gpu = spsolve_gpu_ravel


# ==========================================================================
# Sparse linear solver on CPU
# ==========================================================================

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


# ==========================================================================
# Register sparse linear solvers
# ==========================================================================

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

# @jax.custom_vjp
@partial(jax.custom_vjp, nondiff_argnums=(3,))
def sparse_solve(q, xyz_fixed, loads, structure):
    """
    The sparse linear solver.
    """
    A = force_densities_to_A(q, structure)
    b = force_densities_to_b(q, loads, xyz_fixed, structure)

    return spsolve(A, b)


# ==========================================================================
# Forward and backward passes
# ==========================================================================

def sparse_solve_fwd(q, xyz_fixed, loads, structure):
    """
    Forward pass of the sparse linear solver.

    Call the linear solve and save parameters and solution for the backward pass.
    """
    xk = sparse_solve(q, xyz_fixed, loads, structure)

    return xk, (xk, q, xyz_fixed, loads)


def sparse_solve_bwd(structure, res, g):
    """
    Backward pass of the sparse linear solver.
    """
    xk, q, xyz_fixed, loads = res

    # function that translates parameters into LHS matrix in CSC format
    A = force_densities_to_A(q, structure)

    # Solve adjoint system
    # A.T @ xk_bar = -g
    lam = spsolve(A, g)

    # the implicit constraint function for implicit differentiation
    def residual_fn(params):
        q, xyz_fixed, loads = params
        A = force_densities_to_A(q, structure)
        b = force_densities_to_b(q, loads, xyz_fixed, structure)
        return b - A @ xk

    params = (q, xyz_fixed, loads)

    # Call vjp of residual_fn to compute gradient wrt params
    params_bar = jax.vjp(residual_fn, params)[1](lam)[0]

    return params_bar


# ==========================================================================
# Force density helpers
# ==========================================================================

def force_densities_to_A(q, structure):
    """
    Computes the LHS matrix in CSC format from a vector of force densities.
    """
    index_array = structure.index_array
    diag_indices = structure.diag_indices
    diags = structure.diags

    nondiags_data = -q[index_array.data - 1]
    nondiags = CSC((nondiags_data, index_array.indices, index_array.indptr),
                   shape=index_array.shape)
    diag_fd = diags.T @ q  # sum of force densities for each node

    nondiags.data = nondiags.data.at[diag_indices].set(diag_fd)

    return nondiags


def force_densities_to_b(q, loads, xyz_fixed, structure):
    """
    Computes the RHS matrix in dense format from a vector of force densities.
    """
    c_free = structure.connectivity_free
    c_fixed = structure.connectivity_fixed
    free = structure.nodes_indices_free

    b = loads[free, :] - c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))

    return b


# ==========================================================================
# Sparse matrix helpers
# ==========================================================================

def sparse_blockdiag_matrix(A, num=2, format=CSC):
    """
    Build a block diagonal sparse matrix in the input format by repeating
    a square sparse matrix a prescribed number of times.
    """
    indptr = []
    indices = []
    data = []

    assert num > 1, "Block diagonal matrix must have at least 2 blocks."

    nrows, ncols = A.shape
    assert nrows == ncols, "Supported sparse matrices must be square."

    c_idx = 0
    r_idx = 0
    for i in range(num):
        _indptr = A.indptr + c_idx
        if i > 0:
            _indptr = _indptr[1:]

        indptr.append(_indptr)
        indices.append(A.indices + r_idx)
        data.append(A.data)

        c_idx += jnp.max(A.indptr)
        r_idx += nrows

    indptr = jnp.concatenate(indptr)
    indices = jnp.concatenate(indices)
    data = jnp.concatenate(data)

    return format((data, indices, indptr),
                  shape=(ncols*num, nrows*num))


# ==========================================================================
# Register forward and backward passes to JAX
# ==========================================================================

sparse_solve.defvjp(sparse_solve_fwd, sparse_solve_bwd)
