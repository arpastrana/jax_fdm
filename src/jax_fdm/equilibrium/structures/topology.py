import numpy as np

import jax
import jax.numpy as jnp

import equinox as eqx

from compas.numerical import connectivity_matrix
from compas.numerical import face_matrix
from compas.utilities import pairwise

from jax.experimental.sparse import BCOO

from jax_fdm import DTYPE_INT_NP

from jax_fdm.equilibrium.structures.mixins import IndexingMixins
from jax_fdm.equilibrium.structures.mixins import MeshIndexingMixins


# ==========================================================================
# Graphs
# ==========================================================================

class Graph(eqx.Module, IndexingMixins):
    """
    A graph.
    """
    nodes: np.ndarray
    edges: np.ndarray
    connectivity: jax.Array

    def __init__(self, nodes, edges):
        self.nodes = nodes
        assert edges.shape[1] == 2, "Edges in graph must connect exactly 2 nodes"
        self.edges = edges
        self.connectivity = self._connectivity_matrix()

    @property
    def num_nodes(self):
        """
        The number of nodes.
        """
        return self.nodes.size

    @property
    def num_edges(self):
        """
        The number of edges.
        """
        return self.edges.shape[0]

    def _connectivity_matrix(self):
        """
        The connectivity matrix between edges and nodes.
        """
        edges_indexed = list(self.edges_indexed)

        return jnp.asarray(connectivity_matrix(edges_indexed, "array"))


class GraphSparse(Graph):
    """
    A sparse graph.
    """
    def _connectivity_matrix(self):
        """
        The connectivity matrix between edges and nodes in JAX format.

        Notes
        -----
        This currently is a dense array, but it should be a sparse one.

        How come?

        Currently there is a JAX bug that prevents us from using the
        sparse format with the connectivity matrix:

          C =  BCOO.from_scipy_sparse(self.connectivity_scipy)

        When not using a dense array from the next line, we get the
        following error:

          TypeError: float() argument must be a string or a number, not 'Zero'

        Therefore, we use the connectivity matrix method from the parent
        class, which outputs a dense array.

        However, submatrices connectivity_free and connectivity_fixed
        are correctly initialized and used as sparse matrices.
        """
        return super()._connectivity_matrix()

    @property
    def connectivity_scipy(self):
        """
        The connectivity matrix between edges and nodes in SciPy CSC format.
        """
        # TODO: Refactor GraphSparse to return a JAX sparse matrix instead
        edges_indexed = list(self.edges_indexed)

        return connectivity_matrix(edges_indexed, "csc")


# ==========================================================================
# Mesh
# ==========================================================================

class Mesh(Graph, MeshIndexingMixins):
    """
    A mesh.
    """
    # The faces array can have rows of different lengths. How to handle it?
    # Using a tuple instead of an array?
    vertices: np.ndarray
    faces: np.ndarray
    connectivity_faces: jax.Array

    def __init__(self, vertices, faces, edges=None, **kwargs):

        if edges is None:
            edges = self._edges_from_faces(faces)

        self.vertices = vertices
        assert faces.shape[1] > 2, "Mesh faces must connect at least 3 vertices"

        self.faces = faces
        self.connectivity_faces = self._connectivity_faces_matrix()

        super().__init__(vertices, edges)

    @property
    def num_vertices(self):
        """
        The number of vertices.
        """
        return self.vertices.size

    @property
    def num_faces(self):
        """
        The number of faces.
        """
        return self.faces.shape[0]

    @staticmethod
    def _edges_from_faces(faces):
        """
        The the edges of the mesh.

        Edges have no topological meaning on a mesh and are used only to
        store data.

        The edges are calculated by first looking at all the halfedges of the
        faces of the mesh, and then only storing the unique halfedges.
        """
        halfedges = []
        for face_vertices in faces:
            for u, v in pairwise(np.concatenate((face_vertices, face_vertices[:1]))):
                halfedge = (int(u), int(v))
                halfedges.append(halfedge)

        edges = []
        visited = set()
        for u, v in halfedges:
            if (u, v) in visited or (v, u) in visited:
                continue
            edge = (u, v)
            visited.add(edge)
            edges.append(edge)

        return np.asarray(edges, dtype=DTYPE_INT_NP)

    def _connectivity_faces_matrix(self):
        """
        The connectivity matrix between faces and nodes.
        """
        faces_indexed = list(self.faces_indexed)
        F = face_matrix(faces_indexed, "array")

        return jnp.asarray(F)


class MeshSparse(Mesh, GraphSparse):
    """
    A sparse mesh.
    """
    def _connectivity_faces_matrix(self):
        """
        The connectivity matrix between faces and nodes in sparse format.
        """
        faces_indexed = list(self.faces_indexed)
        F = face_matrix(faces_indexed, "csc")

        return BCOO.from_scipy_sparse(F)


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":

    from compas.datastructures import Network as CNetwork
    from compas.datastructures import Mesh as CMesh
    from compas.utilities import pairwise
    from jax import grad

    num_nodes = 5
    nodes = list(range(num_nodes))
    edges = [edge for edge in pairwise(nodes)]

    nodes = jnp.array(nodes, dtype=jnp.int64)
    edges = jnp.array(edges, dtype=jnp.int64)

    graph = Graph(nodes, edges)
    # supports = np.zeros_like(nodes)
    # supports[0] = 1
    # supports[-1] = 1
    # supports = jnp.asarray(supports)
    # structure = EquilibriumStructure(nodes, edges, supports)
    # assert structure.num_supports == 2
    # print(structure)
    # print(structure.supports)
    # print(structure.nodes_free)
    # print(structure.nodes_fixed)
    # print(structure.nodes_freefixed)

    # print(graph.nodes)
    # print(graph.edges)
    # print(graph.num_nodes)
    # print(graph.num_edges)
    # print(graph.connectivity_matrix)

    graph_sparse = GraphSparse(nodes, edges)

    # print(graph_sparse.nodes)
    # print(graph_sparse.edges)
    # print(graph_sparse.num_nodes)
    # print(graph_sparse.num_edges)

    assert jnp.allclose(graph_sparse.connectivity,
                        graph.connectivity)

    cmesh = CMesh.from_meshgrid(2.0, 2)
    print(cmesh)

    cmesh_faces = [cmesh.face_vertices(fkey) for fkey in cmesh.faces()]
    # cmesh_faces[0].append(-1)

    # print("Compas mesh faces")
    # print(cmesh_faces)
    # print("Compas mesh edges")
    # print(list(cmesh.edges()))

    vertices_array = np.asarray(list(cmesh.vertices()))
    faces_array = np.asarray(cmesh_faces)

    # def f(g):
    #     return jnp.sum(jnp.square(g.nodes - 1.0))

    # y = f(graph)
    # print(y)

    # from jax import jit
    # jf = jit(f)
    # z = jf(graph)
    # assert y == z
    # print(y, z)

    # gjf = jit(grad(f))
    # w = gjf(graph)
    # print("w", w)

    print("All good, cowboy!")
