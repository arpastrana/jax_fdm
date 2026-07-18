"""
Parity tests between the dense and sparse equilibrium pipelines.

The sparse structures keep their connectivity in JAX sparse format while the
dense ones hold plain arrays; both must produce the same equilibrium states.
Loading all three carriers (vertices, edges, faces) with ``tmax > 1`` runs the
full load-assembly pipeline, pinning the index-array rewrite of the tributary
load functions numerically against the dense reference.
"""

import jax.numpy as jnp
import pytest

from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import fdm


@pytest.fixture
def loaded_mesh():
    """
    A saddled meshgrid with force densities and vertex, edge, and face loads.
    """
    mesh = FDMesh.from_meshgrid(dx=2, nx=5)

    for vertex in mesh.vertices_on_boundary():
        mesh.vertex_support(vertex)

    mesh.edges_forcedensities(-2.0)
    mesh.vertices_loads([0.1, -0.2, -0.5])
    mesh.edges_loads([0.0, 0.0, -0.3])
    mesh.faces_loads([0.05, 0.0, -1.0])

    return mesh


def test_fdm_dense_sparse_parity_all_load_types(loaded_mesh):
    """
    Dense and sparse fdm() agree on a mesh carrying all three load types.
    """
    mesh_dense = fdm(loaded_mesh, sparse=False, tmax=100)
    mesh_sparse = fdm(loaded_mesh, sparse=True, tmax=100)

    xyz_dense = jnp.asarray(
        [mesh_dense.vertex_coordinates(v) for v in mesh_dense.vertices()],
    )
    xyz_sparse = jnp.asarray(
        [mesh_sparse.vertex_coordinates(v) for v in mesh_sparse.vertices()],
    )
    assert jnp.allclose(xyz_dense, xyz_sparse, atol=1e-6)

    forces_dense = jnp.asarray(
        [mesh_dense.edge_force(e) for e in mesh_dense.edges()],
    )
    forces_sparse = jnp.asarray(
        [mesh_sparse.edge_force(e) for e in mesh_sparse.edges()],
    )
    assert jnp.allclose(forces_dense, forces_sparse, atol=1e-6)

    residuals_dense = jnp.asarray(
        [mesh_dense.vertex_residual(v) for v in mesh_dense.vertices()],
    )
    residuals_sparse = jnp.asarray(
        [mesh_sparse.vertex_residual(v) for v in mesh_sparse.vertices()],
    )
    assert jnp.allclose(residuals_dense, residuals_sparse, atol=1e-6)
