from functools import partial
import jax
import jax.numpy as jnp
import numpy as np

import scipy.sparse
import scipy.sparse.linalg
from jax.experimental.sparse import BCOO, CSC


def force_densities_to_A(q, index_array, diag_indices, diags):
    """Computes the LHS matrix from a given vector of force densities."""

    nondiags_data = -q[index_array.data]
    nondiags = CSC((nondiags_data, index_array.indices, index_array.indptr), shape=index_array.shape)
    diag_fd = diags.T @ q  # sum of force densities for each node

    nondiags.data = nondiags.data.at[diag_indices].set(diag_fd)
    return nondiags

def linear_solve_callback(data, indices, indptr, b):
    csc_matrix = scipy.sparse.csc_matrix((data, indices, indptr))
    return scipy.sparse.linalg.spsolve(csc_matrix, b)

@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 7, 8))
def linear_solve(q, xyz_fixed, loads, free, c_free, c_fixed, index_array, diag_indices, diags):

    A = force_densities_to_A(q, index_array, diag_indices, diags)
    b = loads[free, :] - c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))

    xk = jax.pure_callback(linear_solve_callback,
                           b,  # return type is b
                           A.data, A.indices, A.indptr, b)  # input arguments
    return xk

def linear_solve_fwd(q, xyz_fixed, loads, free, c_free, c_fixed, index_array, diag_indices, diags):
    xk = linear_solve(q, xyz_fixed, loads, free, c_free, c_fixed, index_array, diag_indices, diags)

    lhs_matrix = force_densities_to_A(q, index_array, diag_indices, diags)
    return xk, (xk, q, xyz_fixed, loads)

def linear_solve_bwd(free, c_free, c_fixed, index_array, diag_indices, diags, res, g):
    xk, q, xyz_fixed, loads = res
    A = force_densities_to_A(q, index_array, diag_indices, diags)

    # Solve adjoint system
    # A.T @ xk_bar = -g
    lam = jax.pure_callback(linear_solve_callback,
                            g,  # return type is g
                            A.data, A.indices, A.indptr, g)  # input arguments

    def residual_fn(params):
        q, xyz_fixed, loads = params
        b = loads[free, :] - c_free.T @ (q[:, None] * (c_fixed @ xyz_fixed))
        A = force_densities_to_A(q, index_array, diag_indices, diags)
        return b - A @ xk

    p = (q, xyz_fixed, loads)
    # Call vjp of residual_fn to compute gradient wrt params
    params_bar = jax.vjp(residual_fn, p)[1](lam)[0]
    return params_bar

linear_solve.defvjp(linear_solve_fwd, linear_solve_bwd)


def get_sparse_diag_indices(csr):
    """Given a CSR matrix, get indices into `data` that access diagonal elements in order."""
    all_indices = []
    for i in range(csr.shape[0]):
        index_range = csr.indices[csr.indptr[i]:csr.indptr[i + 1]]
        ind_loc = np.where(index_range == i)[0]
        all_indices.append(ind_loc + csr.indptr[i])

    return np.concatenate(all_indices)