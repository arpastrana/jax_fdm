import numpy as np
from scipy.sparse import coo_matrix

import jax
import jax.numpy as jnp

import equinox as eqx

from compas.numerical import connectivity_matrix
from compas.numerical import face_matrix as compas_face_matrix
from compas.utilities import pairwise

from jax.experimental.sparse import BCOO

from jax_fdm import DTYPE_NP
from jax_fdm import DTYPE_INT_NP
from jax_fdm import DTYPE_JAX

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
    edges_indexed: jax.Array
    connectivity: jax.Array
    adjacency: jax.Array

    def __init__(self, nodes, edges):
        self.nodes = nodes

        assert edges.shape[1] == 2, "Edges in graph must connect exactly 2 nodes"
        self.edges = edges
        self.edges_indexed = self._edges_indexed()

        self.connectivity = self._connectivity_matrix()
        self.adjacency = self._adjacency_matrix()

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
        edges_indexed = self.edges_indexed

        return jnp.array(connectivity_matrix(edges_indexed, "array"), dtype=DTYPE_JAX)

    def _adjacency_matrix(self):
        """
        The adjacency matrix between nodes and nodes.
        """
        edges_indexed = self.edges_indexed

        return jnp.array(adjacency_matrix(edges_indexed, "array"), dtype=DTYPE_JAX)


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
        # C = super()._connectivity_matrix()
        # return BCOO.fromdense(C).astype(DTYPE_JAX)

        # C = self.connectivity_scipy
        # return BCOO.from_scipy_sparse(C)[:, :]

        # C = self.connectivity_scipy
        # args = (C.data, C.indices, C.indptr)
        # return CSC(args, shape=C.shape)

        return super()._connectivity_matrix()

    @property
    def connectivity_scipy(self):
        """
        The connectivity matrix between edges and nodes in SciPy CSC format.
        """
        # TODO: Refactor GraphSparse to return a JAX sparse matrix instead
        edges_indexed = self.edges_indexed

        return connectivity_matrix(edges_indexed, "csc")

    def _adjacency_matrix(self):
        """
        The adjacency matrix between nodes and nodes.
        """
        edges_indexed = self.edges_indexed

        A = adjacency_matrix(edges_indexed, "coo")

        return BCOO.from_scipy_sparse(A).todense()


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
    faces_indexed: jax.Array
    connectivity_faces_vertices: jax.Array
    connectivity_edges_faces: jax.Array

    def __init__(self, vertices, faces, edges=None, **kwargs):

        if edges is None:
            edges = self._edges_from_faces(faces)

        self.vertices = vertices
        assert faces.shape[1] > 2, "Mesh faces must connect at least 3 vertices"

        self.faces = faces
        self.faces_indexed = self._faces_indexed()
        self.connectivity_faces_vertices = self._connectivity_faces_matrix()

        super().__init__(vertices, edges)

        self.connectivity_edges_faces = self._connectivity_edges_faces_matrix()

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

    def _connectivity_edges_faces_matrix(self):
        """
        The connectivity matrix between edges and faces of a mesh.
        """
        C = np.zeros((self.num_edges, self.num_faces))

        for eindex, findex in enumerate(self._edges_faces()):
            C[eindex, findex] = 1.

        return jnp.asarray(C)

    def _edges_faces(self):
        """
        The connectivity matrix of the edges and the faces of a mesh.
        """
        edges_faces = []
        for u, v in self.edges:

            edge = (int(u), int(v))
            findices = []

            for findex, face in enumerate(self.faces):
                face = [vkey for vkey in face if vkey >= 0]
                face_loop = np.concatenate((face, face[:1]))
                for u, v in pairwise(face_loop):
                    # iterate one time up, one time clockwise, another counter-clockwise
                    halfedge1 = (int(u), int(v))
                    halfedge2 = (int(v), int(u))

                    if edge == halfedge1 or edge == halfedge2:
                        findices.append(findex)

            assert len(findices) <= 2

            edges_faces.append(tuple(findices))

        return tuple(edges_faces)

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
        for face in faces:
            face_loop = np.concatenate((face, face[:1]))
            for u, v in pairwise(face_loop):
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
        F = face_matrix(self.faces_indexed, "array", normalize=True)

        return jnp.asarray(F)


class MeshSparse(Mesh, GraphSparse):
    """
    A sparse mesh.

    Notes
    -----
    The connectivity matrices are currently dense matrices instead of sparse
    (as they should be) because I have encountered issues with the gradient
    computation via backpropagation if I leave them as sparse matrices.

    The error I see is the following:
    "TypeError: float() argument must be a string or a number, not 'Zero'"

    Which presumably arises upon calling bcoo._bcoo_dot_general_transpose.
    """
    def _connectivity_faces_matrix(self):
        """
        The connectivity matrix between faces and nodes in sparse format.
        """
        F = face_matrix(self.faces_indexed, "csc", normalize=True)

        return BCOO.from_scipy_sparse(F).todense()

    def _connectivity_edges_faces_matrix(self):
        """
        The connectivity matrix between edges and faces of a mesh in sparse format.
        """
        C = np.zeros((self.num_edges, self.num_faces))

        for eindex, findex in enumerate(self._edges_faces()):
            C[eindex, findex] = 1.

        return jnp.asarray(C)


# ==========================================================================
# Main
# ==========================================================================

def mesh_edges_faces(mesh):
    """
    The connectivity matrix of the edges and the faces of a mesh.
    """
    face_index = {fkey: idx for idx, fkey in enumerate(mesh.faces())}

    edges_faces = []
    for u, v in mesh.edges():

        findices = []
        for fkey in mesh.edge_faces(u, v):

            if fkey is None:
                continue

            findex = face_index[fkey]
            findices.append(findex)

        assert len(findices) <= 2

        edges_faces.append(tuple(findices))

    return edges_faces


def mesh_connectivity_edges_faces(mesh):
    """
    The connectivity matrix between edges and faces of a mesh.
    """
    num_edges = len(list(mesh.edges()))
    num_faces = mesh.number_of_faces()

    connectivity = np.zeros((num_edges, num_faces))

    edges_faces = mesh_edges_faces(mesh)

    for eindex, findex in enumerate(edges_faces):
        connectivity[eindex, findex] = 1.

    return connectivity


def face_matrix(face_vertices, rtype="array", normalize=False):
    """
    Creates a face-vertex adjacency matrix that skips -1 vertex entries.
    """
    face_vertices_clean = []
    for face in face_vertices:
        face_clean = [vertex for vertex in face if vertex >= 0]
        face_vertices_clean.append(face_clean)

    return compas_face_matrix(face_vertices_clean, rtype, normalize)


def adjacency_matrix(edges, rtype="array"):
    """
    Creates a vertex-vertex adjacency matrix.

    It expects that vertices / nodes are continuously indexed (no skips),
    and that edges are indexed from 0 to len(vertices) / len(nodes).
    """
    num_vertices = np.max(np.ravel(edges)) + 1

    # rows and columns indices for the COO format
    rows = np.hstack([edges[:, 0], edges[:, 1]])  # add edges in both directions for undirected graph
    cols = np.hstack([edges[:, 1], edges[:, 0]])

    # data to fill in (all 1s for the existence of edges)
    data = np.ones(len(rows), dtype=DTYPE_NP)

    # create the COO matrix
    A = coo_matrix(
        (data, (rows, cols)),
        shape=(num_vertices, num_vertices)
    )

    # convert to floating point matrix
    return _return_matrix(A.asfptype(), rtype)


def _return_matrix(M, rtype):
    if rtype == "list":
        return M.toarray().tolist()
    if rtype == "array":
        return M.toarray()
    if rtype == "csr":
        return M.tocsr()
    if rtype == "csc":
        return M.tocsc()
    if rtype == "coo":
        return M.tocoo()
    return M


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":

    from compas.datastructures import Mesh as CMesh
    from compas.numerical import adjacency_matrix as adjacency_matrix_compas

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

    # test connectivity edges faces
    C = mesh_connectivity_edges_faces(cmesh)
    cmesh_faces = [cmesh.face_vertices(fkey) for fkey in cmesh.faces()]
    # cmesh_faces[0].append(-1)

    vertices_array = np.asarray(list(cmesh.vertices()))
    faces_array = np.asarray(cmesh_faces)
    edges_array = np.asarray(list(cmesh.edges()))
    mesh = MeshSparse(vertices_array, faces_array, edges_array)

    jnp.allclose(C, mesh.connectivity_edges_faces)

    # test adjacency matrix
    vertex_index = cmesh.vertex_index()
    adjacency = [[vertex_index[nbr] for nbr in cmesh.vertex_neighbors(vertex)] for vertex in cmesh.vertices()]
    A_c = adjacency_matrix_compas(adjacency, rtype="array")

    A = mesh.adjacency.todense()
    jnp.allclose(A, A_c)

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
