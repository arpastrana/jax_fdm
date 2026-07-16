"""Sparse linear solvers for the FDM system; reverse-mode autodiff only."""

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import CSC
from jax.experimental.sparse.linalg import spsolve as spsolve_jax
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu as splu_scipy
from scipy.sparse.linalg import spsolve as spsolve_scipy

# ==========================================================================
# Type aliases for the general linear system A x = b
# ==========================================================================

# The left-hand side matrix A, right-hand side b (one column per rhs), and
# solution x. The three axes are the general linear-solver axes: `equations`
# (rows of A and b), `unknowns` (columns of A and rows of x), and `rhs` (the
# number of right-hand sides solved at once, e.g. the three xyz coordinates).
SystemMatrixLHS = Float[CSC, "equations unknowns"]
SystemMatrixRHS = Float[Array, "equations rhs"]
SystemSolution = Float[Array, "unknowns rhs"]

# ==========================================================================
# Sparse linear solver on GPU
# ==========================================================================


def spsolve_gpu_ravel(A: SystemMatrixLHS, b: SystemMatrixRHS) -> SystemSolution:
    """
    Solve the sparse system on GPU by raveling all right-hand sides into one.

    Parameters
    ----------
    A :
        The left-hand side matrix, assumed symmetric.
    b :
        The right-hand side, one column per coordinate.

    Returns
    -------
    x :
        The solution, one column per right-hand side.

    Notes
    -----
    The CUDA sparse backend cannot take a matrix right-hand side, so ``b`` is
    raveled into one vector and ``A`` is repeated into a block-diagonal matrix (one
    block per coordinate). This solves the whole block system at once, which can be
    faster than :func:`spsolve_gpu_stack`, though the CUDA limitation may still make
    a GPU solve costlier than a CPU one. So use this method with a pinch of salt.
    """
    # NOTE: we can pass CSC indices directly to the JAX solver because we can!
    # Just kidding. This is because the matrix A is symmetric :)
    A = blockdiag_matrix_sparse(A, 3)
    b = jnp.ravel(b, "F")
    X = spsolve_jax(A.data, A.indices, A.indptr, b)

    return jnp.reshape(X, (3, -1)).T


def spsolve_gpu_stack(A: SystemMatrixLHS, b: SystemMatrixRHS) -> SystemSolution:
    """
    Solve the sparse system on GPU one right-hand side column at a time.

    Parameters
    ----------
    A :
        The left-hand side matrix, assumed symmetric.
    b :
        The right-hand side, one column per coordinate.

    Returns
    -------
    x :
        The solution, one column per right-hand side.

    Notes
    -----
    The CUDA sparse backend only accepts vector right-hand sides, so the three
    coordinate columns are solved separately and stacked. This can be up to three
    times costlier than solving them together, as :func:`spsolve_gpu_ravel` does.
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


def spsolve_cpu(A: SystemMatrixLHS, b: SystemMatrixRHS) -> SystemSolution:
    """
    Solve the sparse system on CPU via SciPy, wrapped as a JAX pure callback.

    Parameters
    ----------
    A :
        The left-hand side matrix.
    b :
        The right-hand side, one column per coordinate.

    Returns
    -------
    x :
        The solution, one column per right-hand side.
    """

    def callback(data, indices, indptr, _b):
        _A = csc_matrix((data, indices, indptr))
        return spsolve_scipy(_A, _b)

    xk = jax.pure_callback(
        callback,  # callback function
        b,  # return type is b
        A.data,  # callback function arguments from here on
        A.indices,
        A.indptr,
        b,
    )
    return xk


# ==========================================================================
# Register sparse linear solvers
# ==========================================================================


def register_sparse_solver(solvers: dict[str, Callable]) -> Callable:
    """
    Pick the sparse solver matching the active JAX backend.

    Parameters
    ----------
    solvers :
        A mapping from backend name (``"cpu"`` or ``"gpu"``) to solver function.

    Returns
    -------
    solver :
        The solver registered for the current default backend.

    Raises
    ------
    ValueError
        If no solver is registered for the active backend.
    """
    backend = jax.default_backend()
    sparse_solver = solvers.get(backend)

    if not sparse_solver:
        raise ValueError(f"Default backend {backend} does not support a sparse solver")

    return sparse_solver


solvers = {"cpu": spsolve_cpu, "gpu": spsolve_gpu}

spsolve = register_sparse_solver(solvers)


# ==========================================================================
# Define sparse linear solver
# ==========================================================================


@jax.custom_vjp
def sparse_solve(A: SystemMatrixLHS, b: SystemMatrixRHS) -> SystemSolution:
    """
    Solve the sparse linear system with a custom-differentiable solver.

    Parameters
    ----------
    A :
        The left-hand side matrix, assumed symmetric.
    b :
        The right-hand side, one column per coordinate.

    Returns
    -------
    x :
        The solution, one column per right-hand side.

    Notes
    -----
    Dispatches to the backend solver registered by :func:`register_sparse_solver`.
    A custom VJP supplies the backward pass, so only reverse-mode differentiation
    is supported.
    """
    return spsolve(A, b)


# ==========================================================================
# Forward and backward passes
# ==========================================================================

# Auxiliary data saved by the forward pass for the backward pass: the
# solution, the left-hand side matrix, and the right-hand side, in that order.
SparseSolveResidual = tuple[SystemSolution, SystemMatrixLHS, SystemMatrixRHS]


def sparse_solve_fwd(
    A: SystemMatrixLHS,
    b: SystemMatrixRHS,
) -> tuple[SystemSolution, SparseSolveResidual]:
    """
    Run the forward pass of the differentiable sparse solve.

    Parameters
    ----------
    A :
        The left-hand side matrix, assumed symmetric.
    b :
        The right-hand side, one column per coordinate.

    Returns
    -------
    solution :
        The solve result together with the residual (solution, ``A``, ``b``) saved
        for the backward pass.
    """
    xk = sparse_solve(A, b)

    return xk, (xk, A, b)


def sparse_solve_bwd(
    res: SparseSolveResidual,
    g: SystemMatrixRHS,
) -> tuple[SystemMatrixLHS, SystemMatrixRHS]:
    """
    Run the backward pass of the differentiable sparse solve.

    Parameters
    ----------
    res :
        The residual saved by the forward pass: the solution, ``A``, and ``b``.
    g :
        The cotangent of the solution.

    Returns
    -------
    grads :
        The cotangents with respect to ``A`` and ``b``.

    Notes
    -----
    Differentiates implicitly through the linear system rather than through the
    solver. The adjoint system reuses ``A`` directly because it is symmetric.
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


def blockdiag_matrix_sparse(
    A: SystemMatrixLHS,
    num: int = 2,
    format: type[CSC] = CSC,
) -> Float[CSC, "equations_blocks unknowns_blocks"]:
    """
    Repeat a square sparse matrix along the diagonal into a block matrix.

    Parameters
    ----------
    A :
        The square sparse matrix to repeat.
    num :
        The number of diagonal blocks; must be at least two.
    format :
        The sparse matrix class of the result.

    Returns
    -------
    block_matrix :
        The block-diagonal matrix with ``num`` copies of ``A`` on its diagonal.

    Raises
    ------
    AssertionError
        If fewer than two blocks are requested, or if ``A`` is not square.
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

    return format((data, indices, indptr), shape=(ncols * num, nrows * num))


# ==========================================================================
# Register forward and backward passes to JAX
# ==========================================================================

sparse_solve.defvjp(sparse_solve_fwd, sparse_solve_bwd)


# ==========================================================================
# Sparse LU factorization on CPU with simple caching
# ==========================================================================

# Simple single-entry cache for SuperLU factorization
# Structure: {'factorization': SuperLU_object, 'session_id': int}
_SPLU_CACHE = {"factorization": None, "session_id": 0}


def splu_clear() -> None:
    """
    Reset the cached sparse LU factorization to free memory.
    """
    _SPLU_CACHE["session_id"] = 0
    _SPLU_CACHE["factorization"] = None


def splu_cpu(A: SystemMatrixLHS, session_id: int | None = None) -> Int[Array, ""]:
    """
    Factorize the sparse matrix on CPU via SciPy, caching the factorization.

    Parameters
    ----------
    A :
        The sparse matrix to factorize.
    session_id :
        The session identifier keying the cache. If None, uses session ``0``.

    Returns
    -------
    session_id :
        The session identifier the factorization was stored under.

    Notes
    -----
    SuperLU objects cannot flow through JAX tracing, so the factorization is kept
    in a single-entry global cache; :func:`splu_solve_cpu` retrieves it by session
    id. Only the matrix data may change between factorizations, not its sparsity
    structure. Because it mutates the global cache, this callback is impure and must
    not be transformed directly with ``grad`` or ``vmap``.
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
        _SPLU_CACHE["factorization"] = _A_fact
        _SPLU_CACHE["session_id"] = sid

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
        jnp.array(session_id, dtype=jnp.int64),
    )

    return sid


def splu_solve_cpu(session_id: Int[Array, ""], b: SystemMatrixRHS) -> SystemSolution:
    """
    Solve the sparse system on CPU from a cached LU factorization.

    Parameters
    ----------
    session_id :
        The session identifier returned by :func:`splu_cpu`, keying the cached
        factorization to reuse.
    b :
        The right-hand side, one column per coordinate.

    Returns
    -------
    x :
        The solution, one column per right-hand side.

    Raises
    ------
    ValueError
        If the cached session id does not match, or if no factorization is cached.
    """

    def callback(sid, _b):
        # Convert to numpy
        sid = int(np.asarray(sid))
        _b = np.asarray(_b)

        # Check that we're using the correct session
        if _SPLU_CACHE["session_id"] != sid:
            raise ValueError(
                f"Session mismatch: expected {sid}, "
                f"but cache has {_SPLU_CACHE['session_id']}. "
                "Make sure to call splu_cpu before splu_solve_cpu in the same session.",
            )

        # Retrieve factorization from cache
        _A_fact = _SPLU_CACHE["factorization"]
        if _A_fact is None:
            raise ValueError("No factorization found in cache. Call splu_cpu first.")

        # Solve and return
        return _A_fact.solve(_b)

    xk = jax.pure_callback(callback, b, session_id, b)

    return xk
