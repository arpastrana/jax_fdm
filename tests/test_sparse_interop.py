"""
Characterization tests for JAX sparse-array interop.

The sparse structures once densified their connectivity matrices after a
sparse build, citing a historical JAX bug where reverse-mode gradients through
a module-held BCOO raised ``TypeError: float() argument must be a string or a
number, not 'Zero'``. The first test proves that bug is gone on the installed
JAX, guarding the precondition for keeping the matrices sparse. The remaining
tests are sentinels pinning the ops JAX sparse arrays still lack; the library
consumers that needed them were rewritten to index arrays and matmuls during
the sparse migration. When a JAX upgrade makes a sentinel fail, the op became
available and the corresponding workaround can be reconsidered.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import vmap
from jax.experimental.sparse import BCOO
from scipy.sparse import coo_matrix

from jax_fdm.equilibrium.structures.graphs import connectivity_matrix


def _line_graph_bcoo(num_edges=6):
    """
    Build the sparse incidence matrix of a line graph as a BCOO array.
    """
    edges = jnp.asarray([(i, i + 1) for i in range(num_edges)])

    return BCOO.from_scipy_sparse(connectivity_matrix(edges))


class _Holder(eqx.Module):
    """
    A minimal equinox module holding a sparse matrix, as the structures would.
    """

    C: BCOO


def test_grad_through_module_held_bcoo_no_zero_error():
    """
    Reverse-mode grad through a module-held BCOO no longer raises the 'Zero' bug.

    This is the exact pattern the sparse structures avoid by densifying: the
    BCOO leaf gets a symbolic Zero cotangent because grad is taken w.r.t. other
    arguments only, which crashed ``bcoo._bcoo_dot_general_transpose`` on old
    JAX versions.
    """
    holder = _Holder(_line_graph_bcoo())
    num_edges, num_nodes = holder.C.shape
    xyz = jnp.asarray(np.random.default_rng(0).random((num_nodes, 3)))
    q = jnp.ones(num_edges)

    def loss(q, xyz):
        vectors = holder.C @ xyz
        residuals = holder.C.T @ (q[:, None] * vectors)
        return jnp.sum(residuals**2)

    grad_q, grad_xyz = jax.grad(loss, argnums=(0, 1))(q, xyz)
    assert grad_q.shape == (num_edges,)
    assert grad_xyz.shape == (num_nodes, 3)
    assert jnp.all(jnp.isfinite(grad_q))
    assert jnp.all(jnp.isfinite(grad_xyz))

    # the same under jit, as the solver runs it
    grad_q_jit, _ = jax.jit(jax.grad(loss, argnums=(0, 1)))(q, xyz)
    assert jnp.allclose(grad_q_jit, grad_q)


def test_sentinel_jnp_abs_rejects_bcoo():
    """
    Sentinel: ``jnp.abs`` on a BCOO still raises.

    `nodes_tributary_edges_load` once computed
    ``jnp.abs(structure.connectivity)`` and was rewritten as a scatter-add over
    ``edges_indexed`` to work with sparse connectivity. If this test fails
    because the op now works, the matmul formulation is available again.
    """
    C = _line_graph_bcoo()

    with pytest.raises(TypeError):
        # the sentinel exercises exactly the call the stubs reject: jnp.abs
        # is not typed for BCOO because it is not supported for BCOO
        jnp.abs(C)  # pyright: ignore[reportArgumentType]


def test_sentinel_vmap_over_unbatched_bcoo_rows_rejects():
    """
    Sentinel: vmap with ``in_axes=0`` over an unbatched BCOO still raises.

    The face-load pipeline once vmapped over rows of the connectivity matrices
    and was rewritten to vmap over index arrays instead. A BCOO built with
    ``n_batch=1`` already supports this today; this sentinel tracks the
    default unbatched layout the structures hold.
    """
    C = _line_graph_bcoo()
    xyz = jnp.ones((C.shape[1], 3))

    with pytest.raises(ValueError):
        vmap(lambda row, x: row @ x, in_axes=(0, None))(C, xyz)


def test_sentinel_bcoo_comparison_rejects():
    """
    Sentinel: elementwise comparison on a BCOO still raises.

    The vertex normal goal once masked incident faces by comparing a column of
    the face-vertex connectivity against zero and was rewritten to compare the
    face topology index array against the vertex index instead. If this test
    fails because the op now works, sparse-matrix masking is available again.
    """
    F = BCOO.from_scipy_sparse(coo_matrix(np.eye(4)))

    with pytest.raises(NotImplementedError):
        jnp.where(F[:, 1] > 0.0, 1.0, 0.0)
