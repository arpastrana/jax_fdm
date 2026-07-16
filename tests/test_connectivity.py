"""
Characterization tests for graph and mesh connectivity matrices.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from compas.matrices import adjacency_matrix as adjacency_matrix_compas
from jax_fdm import DTYPE_INT_NP
from jax_fdm.equilibrium import mesh_connectivity_edges_faces
from jax_fdm.equilibrium.structures.graphs import Graph
from jax_fdm.equilibrium.structures.graphs import GraphSparse
from jax_fdm.equilibrium.structures.meshes import Mesh
from jax_fdm.equilibrium.structures.meshes import MeshSparse


def _dense(matrix):
    """
    Return a dense array whether the matrix is already dense or a JAX sparse one.
    """
    return matrix.todense() if hasattr(matrix, "todense") else matrix


def _mesh_arrays(cmesh):
    """
    Extract vertex, face, and edge index arrays from a COMPAS mesh.
    """
    faces = [cmesh.face_vertices(fkey) for fkey in cmesh.faces()]
    vertices = np.asarray(list(cmesh.vertices()), dtype=DTYPE_INT_NP)
    faces = np.asarray(faces, dtype=DTYPE_INT_NP)
    edges = np.asarray(list(cmesh.edges()), dtype=DTYPE_INT_NP)

    return vertices, faces, edges


# ==============================================================================
# Graph
# ==============================================================================


def test_graph_dense_sparse_connectivity_agree():
    """
    The dense and sparse graphs build the same connectivity matrix.
    """
    nodes = np.arange(5, dtype=DTYPE_INT_NP)
    edges = np.array([(i, i + 1) for i in range(4)], dtype=DTYPE_INT_NP)

    graph = Graph(nodes, edges)
    graph_sparse = GraphSparse(nodes, edges)

    assert jnp.allclose(_dense(graph_sparse.connectivity), _dense(graph.connectivity))
    assert jnp.allclose(_dense(graph_sparse.adjacency), _dense(graph.adjacency))


# ==============================================================================
# Mesh
# ==============================================================================


def test_mesh_dense_sparse_connectivity_agree(meshgrid_mesh):
    """
    The dense and sparse meshes build the same connectivity and adjacency.
    """
    vertices, faces, edges = _mesh_arrays(meshgrid_mesh)

    mesh = Mesh(vertices, faces, edges)
    mesh_sparse = MeshSparse(vertices, faces, edges)

    assert jnp.allclose(_dense(mesh_sparse.connectivity), _dense(mesh.connectivity))
    assert jnp.allclose(
        _dense(mesh_sparse.connectivity_edges_faces),
        _dense(mesh.connectivity_edges_faces),
    )
    assert jnp.allclose(_dense(mesh_sparse.adjacency), _dense(mesh.adjacency))


@pytest.mark.compas_xcheck
def test_mesh_connectivity_matches_compas(meshgrid_mesh):
    """
    The array-built mesh connectivity matches COMPAS halfedge traversal.
    """
    vertices, faces, edges = _mesh_arrays(meshgrid_mesh)
    mesh = MeshSparse(vertices, faces, edges)

    # jax_fdm derives edge-face adjacency from numpy arrays; the module helper
    # walks the COMPAS halfedge topology instead.
    connectivity_compas = mesh_connectivity_edges_faces(meshgrid_mesh)
    assert jnp.allclose(connectivity_compas, _dense(mesh.connectivity_edges_faces))

    vertex_index = meshgrid_mesh.vertex_index()
    adjacency = [
        [vertex_index[nbr] for nbr in meshgrid_mesh.vertex_neighbors(vertex)]
        for vertex in meshgrid_mesh.vertices()
    ]
    adjacency_compas = adjacency_matrix_compas(adjacency, rtype="array")
    assert jnp.allclose(_dense(mesh.adjacency), adjacency_compas)
