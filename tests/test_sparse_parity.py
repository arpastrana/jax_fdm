"""
Parity tests between the dense and sparse equilibrium pipelines.

The sparse structures keep their connectivity in JAX sparse format while the
dense ones hold plain arrays; both must produce the same equilibrium states.
Loading all three carriers (vertices, edges, faces) with ``tmax > 1`` runs the
full load-assembly pipeline, pinning the index-array rewrite of the tributary
load functions numerically against the dense reference. The goal parity test
pins the matmul rewrite of the neighborhood goals against both structure
variants, whichever storage format each keeps its matrices in.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_fdm.datastructures import FDMesh
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import fdm
from jax_fdm.equilibrium.fdm import structure_from_datastructure
from jax_fdm.goals import MeshSmoothGoal
from jax_fdm.goals import MeshXYZFaceLaplacianGoal
from jax_fdm.goals import VertexNormalAngleGoal


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


def _goal_predictions(mesh, sparse):
    """
    Evaluate the neighborhood goal predictions on a fixed random state.
    """
    structure = structure_from_datastructure(mesh, sparse=sparse)

    num_nodes = structure.num_nodes
    num_edges = structure.num_edges
    rng = np.random.default_rng(7)
    xyz = jnp.asarray(rng.random((num_nodes, 3)) * 3.0)
    eq_state = EquilibriumState(
        xyz=xyz,
        residuals=jnp.zeros((num_nodes, 3)),
        lengths=jnp.zeros((num_edges, 1)),
        forces=jnp.zeros((num_edges, 1)),
        loads=jnp.zeros((num_nodes, 3)),
        vectors=jnp.zeros((num_edges, 3)),
    )

    sentinel = jnp.asarray([0])

    smooth_goal = MeshSmoothGoal()
    laplacian_goal = MeshXYZFaceLaplacianGoal()
    normal_goal = VertexNormalAngleGoal(key=12, vector=[0.0, 0.0, 1.0], target=0.0)

    return (
        smooth_goal.prediction(eq_state, structure, sentinel),
        laplacian_goal.laplacian_vertices(eq_state, structure),
        normal_goal.vertex_normal(eq_state, structure.faces_indexed, jnp.asarray(12)),
    )


def test_neighborhood_goals_dense_sparse_parity(loaded_mesh):
    """
    The matmul-formulated goals agree between the dense and sparse structures.

    The smoothing, face-laplacian, and vertex-normal goals read the adjacency,
    face-vertex, and face topology arrays off the structure, whichever storage
    format the structure keeps them in.
    """
    preds_dense = _goal_predictions(loaded_mesh, sparse=False)
    preds_sparse = _goal_predictions(loaded_mesh, sparse=True)

    for pred_dense, pred_sparse in zip(preds_dense, preds_sparse):
        assert jnp.allclose(pred_dense, pred_sparse, atol=1e-9)
