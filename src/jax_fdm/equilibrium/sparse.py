"""
NOTE: Sparse solver does not support forward mode auto-differentiation yet.
"""
import jax
import jax.numpy as jnp
import numpy as np

from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve as spsolve_scipy
from scipy.sparse.linalg import splu as splu_scipy

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
    # NOTE: No need to transpose A because it is assumed symmetric
    lam = sparse_solve(A, g)

    # The implicit constraint function for implicit differentiation
    def residual_fn(params):
        A, b = params
        return b - A @ xk

    params = (A, b)

    # Call vjp of residual_fn to compute gradient wrt parameters
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


# ==========================================================================
# Sparse LU factorization on CPU with simple caching
# ==========================================================================

# Simple single-entry cache for SuperLU factorization
# Structure: {'factorization': SuperLU_object, 'session_id': int}
_SPLU_CACHE = {'factorization': None, 'session_id': 0}


def splu_clear():
    """
    Clear the SPLU cache to free memory.
    """
    _SPLU_CACHE['session_id'] = 0
    _SPLU_CACHE['factorization'] = None


def splu_cpu(A, session_id=None):
    """
    A wrapper around scipy sparse LU factorization that acts as a JAX pure callback.

    Parameters
    ----------
    A : CSC matrix
        The sparse matrix to factorize.
    session_id : int, optional
        Session identifier. If None, uses a default session (0).

    Notes
    -----
    Since SuperLU objects cannot be passed through JAX's tracing system, we store
    the factorization in a simple global cache. Only one factorization is kept at
    a time per session. The structure (indices, indptr) must remain constant; only
    the data values may change between factorizations.

    Note that this function modifies the global cache in-place, so it violates
    the JAX function purity guarantee. Therefore, we only use it in functions
    that need not be transformed directly with grad or vmap.
    """
    if session_id is None:
        session_id = 0

    def callback(data, indices, indptr, shape, sid):
        # Convert to numpy arrays
        data = np.asarray(data)
        indices = np.asarray(indices)
        indptr = np.asarray(indptr)
        shape = tuple(np.asarray(shape).astype(int))
        sid = int(np.asarray(sid))

        # Create matrix and factorize
        _A = csc_matrix((data, indices, indptr), shape=shape)
        _A_fact = splu_scipy(_A)

        # Store factorization (overwrite any previous)
        _SPLU_CACHE['factorization'] = _A_fact
        _SPLU_CACHE['session_id'] = sid

        # Return the session ID
        return np.array(sid, dtype=np.int64)

    # Return type is a scalar int64 (the session ID)
    sid = jax.pure_callback(
        callback,
        jax.ShapeDtypeStruct((), jnp.int64),
        A.data,
        A.indices,
        A.indptr,
        jnp.array(A.shape),
        jnp.array(session_id, dtype=jnp.int64)
    )

    return sid


def splu_solve_cpu(session_id, b):
    """
    A wrapper around scipy sparse LU solve that acts as a JAX pure callback.

    Parameters
    ----------
    session_id : scalar int64
        The session ID returned by splu_cpu.
    b : array
        The right-hand side vector or matrix.
    """
    def callback(sid, _b):
        # Convert to numpy
        sid = int(np.asarray(sid))
        _b = np.asarray(_b)

        # Check that we're using the correct session
        if _SPLU_CACHE['session_id'] != sid:
            raise ValueError(
                f"Session mismatch: expected {sid}, but cache has {_SPLU_CACHE['session_id']}. "
                "Make sure to call splu_cpu before splu_solve_cpu in the same session."
            )

        # Retrieve factorization from cache
        _A_fact = _SPLU_CACHE['factorization']
        if _A_fact is None:
            raise ValueError("No factorization found in cache. Call splu_cpu first.")

        # Solve and return
        return _A_fact.solve(_b)

    xk = jax.pure_callback(callback, b, session_id, b)

    return xk
