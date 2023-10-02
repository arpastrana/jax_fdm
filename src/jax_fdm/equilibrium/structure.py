import numpy as np

import jax
import jax.numpy as jnp

import equinox as eqx

from compas.numerical import connectivity_matrix

from jax.experimental.sparse import BCOO
from jax.experimental.sparse import CSC

from jax_fdm import DTYPE_INT_JAX
from jax_fdm import DTYPE_INT_NP


# ==========================================================================
# Mixins
# ==========================================================================

class IndexingMixins:
    @property
    def node_index(self):
        """
        A dictionary between node keys and their enumeration indices.
        """
        return {node.item(): index for index, node in enumerate(self.nodes)}

    @property
    def edge_index(self):
        """
        A dictionary between edge keys and their enumeration indices.
        """
        return {(u.item(), v.item()): index for index, (u, v) in enumerate(self.edges)}

    @property
    def edges_indexed(self):
        """
        An iterable with the edge pointing to the indices of the node keys.
        """
        node_index = self.node_index

        for u, v in self.edges:
            yield node_index[u.item()], node_index[v.item()]


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
# Structure
# ==========================================================================

class EquilibriumStructure(Graph):
    """
    A structure.
    """
    supports: np.ndarray

    connectivity_free: jax.Array
    connectivity_fixed: jax.Array

    nodes_indices_free: jax.Array
    nodes_indices_fixed: jax.Array
    nodes_indices_freefixed: jax.Array

    def __init__(self, nodes, edges, supports):
        super().__init__(nodes, edges)

        self.supports = supports

        self.nodes_indices_free = self._nodes_indices_free()
        self.nodes_indices_fixed = self._nodes_indices_fixed()
        self.nodes_indices_freefixed = self._nodes_indices_freefixed()

        self.connectivity_free = self._connectivity_free()
        self.connectivity_fixed = self._connectivity_fixed()

    @classmethod
    def from_network(cls, network):
        """
        Create a structure from a force density network.
        """
        nodes = list(network.nodes())
        edges = list(network.edges())

        supports = []
        for node in nodes:
            flag = 0.0
            if network.is_node_support(node):
                flag = 1.0
            supports.append(flag)

        nodes = np.asarray(nodes, dtype=DTYPE_INT_NP)
        edges = np.asarray(edges, dtype=DTYPE_INT_NP)
        supports = np.asarray(supports, dtype=DTYPE_INT_NP)

        return cls(nodes, edges, supports)

    @property
    def num_supports(self):
        """
        The number of supports.
        """
        return jnp.count_nonzero(self.supports)

    @property
    def num_free(self):
        """
        The number of supports.
        """
        return self.num_nodes - self.num_supports

    def _connectivity_free(self):
        """
        The connectivity matrix between edges and nodes.
        """
        return self.connectivity[:, self.nodes_indices_free]

    def _connectivity_fixed(self):
        """
        The connectivity matrix between edges and nodes.
        """
        return self.connectivity[:, self.nodes_indices_fixed]

    def _nodes_indices_free(self):
        """
        The indices of the unsupported nodes in the structure.
        """
        indices = jnp.flatnonzero(self.supports == 0, size=self.num_free)

        return indices

    def _nodes_indices_fixed(self):
        """
        The indices of the unsupported nodes in the structure.
        """
        indices = jnp.flatnonzero(self.supports, size=self.num_supports)

        return indices

    def _nodes_indices_freefixed(self):
        """
        A list with the node keys of all the nodes sorted by their node index.
        """
        # TODO: this method must be refactored to be more transparent.
        freefixed_indices = jnp.concatenate([self.nodes_indices_free,
                                             self.nodes_indices_fixed])

        indices = {node.item(): index for index, node in enumerate(freefixed_indices)}
        sorted_indices = []
        for _, index in sorted(indices.items(), key=lambda item: item[0]):
            sorted_indices.append(index)

        return jnp.asarray(sorted_indices, dtype=DTYPE_INT_JAX)


# ==========================================================================
# Sparse structure
# ==========================================================================

class EquilibriumStructureSparse(EquilibriumStructure, GraphSparse):
    """
    A sparse structure.
    """
    index_array: jax.Array
    diag_indices: jax.Array
    diags: jax.Array

    def __init__(self, nodes, edges, supports):
        super().__init__(nodes, edges, supports)

        # Do some precomputation to be able to construct
        # the lhs matrix through indexing
        c_free_csc = self.connectivity_scipy[:, self.nodes_indices_free]
        index_array = self._get_sparse_index_array(c_free_csc)
        self.index_array = index_array

        # Indices of data corresponding to diagonal.
        # With this array we can just index directly into the
        # CSC.data array to refer to the diagonal entries.
        self.diag_indices = self._get_sparse_diag_indices(index_array)

        # Prepare the array D st when D.T @ q we get the diagonal elements of matrix
        self.diags = self._get_sparse_diag_data(c_free_csc)

    # @property
    def _connectivity_free(self):
        """
        The connectivity matrix between edges and nodes.
        """
        return BCOO.from_scipy_sparse(self.connectivity_scipy[:, self.nodes_indices_free])

    # @property
    def _connectivity_fixed(self):
        """
        The connectivity matrix between edges and nodes.
        """
        return BCOO.from_scipy_sparse(self.connectivity_scipy[:, self.nodes_indices_fixed])

    @staticmethod
    def _get_sparse_index_array(c_free_csc):
        """
        Create an index array such that the off-diagonals can index into the
        force density vector.

        This array is used to create the off-diagonal entries of the lhs matrix.

        # NOTE: The input matrix must be a scipy sparse array!
        """
        fd_mod_c_free_csc = c_free_csc.copy()
        fd_mod_c_free_csc.data *= np.take(np.arange(c_free_csc.shape[0]) + 1,
                                          c_free_csc.indices)
        index_array = -(c_free_csc.T @ fd_mod_c_free_csc)

        # The diagonal entries should be set to 0 so that it indexes
        # into a valid entry, but will later be overwritten.
        index_array.setdiag(0)
        index_array = index_array.astype(int)

        return CSC((index_array.data, index_array.indices, index_array.indptr),
                   shape=index_array.shape)

    @staticmethod
    def _get_sparse_diag_indices(csc):
        """
        Given a CSC matrix, get indices into `data` that access diagonal elements in order.
        """
        all_indices = []
        for i in range(csc.shape[0]):
            index_range = csc.indices[csc.indptr[i]:csc.indptr[i + 1]]
            ind_loc = jnp.where(index_range == i)[0]
            all_indices.append(ind_loc + csc.indptr[i])

        return jnp.concatenate(all_indices)

    @staticmethod
    def _get_sparse_diag_data(c_free_csc):
        """
        The diagonal of the lhs matrix is the sum of force densities for
        each outgoing/incoming edge on the node.

        We create the `diags` matrix such that when we multiply it with the
        force density vector we get the diagonal.
        """
        diags_data = jnp.ones_like(c_free_csc.data)

        return CSC((diags_data, c_free_csc.indices, c_free_csc.indptr),
                   shape=c_free_csc.shape)


# ==========================================================================
# Mesh
# ==========================================================================

# class Mesh(Graph):
#     nodes: jax.Array
#     faces: jax.Array
#     edges: jax.Array

#     def __init__(self, nodes, faces, edges):
#         self.nodes = nodes
#         self.faces = faces
#         self.edges = edges
#         # self.edges = self._edges_from_faces(faces)

#     @property
#     def num_faces(self):
#         """
#         The number of faces.
#         """
#         return self.faces.shape[0]

#     @property
#     def face_matrix(self):
#         """
#         The connectivity matrix between faces and nodes.
#         """
#         return jnp.asarray(face_matrix(self.faces, "array"), dtype=DTYPE_JAX)

#     @staticmethod
#     def _edges_from_faces(faces):
#         """
#         The the edges of the mesh.

#         Edges have no topological meaning on a mesh and are used only to
#         store data.
#         The edges are calculated by first looking at all the halfedges of the
#         faces of the mesh, and then only storing the unique halfedges.
#         """
#         # NOTE: This method is producing results that do not match COMPAS'
#         halfedges = []
#         for face_vertices in faces:
#             for u, v in pairwise(face_vertices + face_vertices[:1]):
#                 halfedge = (u.item(), v.item())
#                 halfedges.append(halfedge)

#         edges = []
#         visited = set()
#         for u, v in halfedges:
#             if (u, v) in visited or (v, u) in visited:
#                 continue
#             edge = (u, v)
#             visited.add(edge)
#             edges.append(list(edge))

#         return jnp.asarray(edges, dtype=DTYPE_INT_JAX)


# class MeshSparse(Mesh):
#     @property
#     def face_matrix(self):
#         """
#         The connectivity matrix between faces and nodes.
#         """
#         F = connectivity_matrix(self.faces, "csc")

#         return BCOO.from_scipy_sparse(F)


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
    supports = np.zeros_like(nodes)
    print(nodes)
    print(supports)
    supports[0] = 1
    supports[-1] = 1
    supports = jnp.asarray(supports)
    print(supports)
    structure = EquilibriumStructure(nodes,
                                     edges,
                                     supports)
    print(structure)
    print(structure.supports)
    assert structure.num_supports == 2
    print(structure.nodes_free)
    print(structure.nodes_fixed)
    print(structure.nodes_freefixed)

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

    assert jnp.allclose(graph_sparse.connectivity_matrix.todense(),
                        graph.connectivity_matrix)

    # cmesh = CMesh.from_meshgrid(2.0, 2)
    # print(cmesh)

    # cmesh_faces = [cmesh.face_vertices(fkey) for fkey in cmesh.faces()]
    # cmesh_faces[0].append(-1)
    # mesh = Mesh(jnp.asarray(list(cmesh.vertices())),
    #             jnp.asarray(cmesh_faces),
    #             jnp.asarray(list(cmesh.edges())))

    # print(mesh)

    # print(mesh.edges)
    # print(list(cmesh.edges()))


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
