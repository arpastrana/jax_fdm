"""
NOTE: Sparse solver does not support forward mode auto-differentiation yet.
"""
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
    A = blockdiag_matrix_sparse(A, 3)
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

@jax.custom_vjp
def sparse_solve(A, b):
    """
    The sparse linear solver.
    """
    return spsolve(A, b)


# ==========================================================================
# Forward and backward passes
# ==========================================================================

def sparse_solve_fwd(A, b):
    """
    Forward pass of the sparse linear solver.

    Call the linear solve and save parameters and solution for the backward pass.
    """
    xk = sparse_solve(A, b)

    return xk, (xk, A, b)


def sparse_solve_bwd(res, g):
    """
    Backward pass of the sparse linear solver.
    """
    xk, A, b = res

    # Solve adjoint system
    # A.T @ xk_bar = -g
    lam = sparse_solve(A, g)

    # the implicit constraint function for implicit differentiation
    def residual_fn(params):
        A, b = params
        return b - A @ xk

    params = (A, b)

    # Call vjp of residual_fn to compute gradient wrt params
    params_bar = jax.vjp(residual_fn, params)[1](lam)[0]

    return params_bar


# ==========================================================================
# Sparse matrix helpers
# ==========================================================================

def blockdiag_matrix_sparse(A, num=2, format=CSC):
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
