import numpy as np

import jax
import jax.numpy as jnp

from jax.experimental.sparse import BCOO
from jax.experimental.sparse import CSC

from jax_fdm import DTYPE_INT_JAX
from jax_fdm import DTYPE_INT_NP

from jax_fdm.equilibrium.structures.graphs import Graph
from jax_fdm.equilibrium.structures.graphs import GraphSparse

from jax_fdm.equilibrium.structures.meshes import Mesh
from jax_fdm.equilibrium.structures.meshes import MeshSparse


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

    indices_free: jax.Array
    indices_fixed: jax.Array
    indices_freefixed: jax.Array

    def __init__(self, nodes, edges, supports, **kwargs):
        super().__init__(nodes=nodes, edges=edges, **kwargs)

        self.supports = supports

        self.indices_free = self._indices_free()
        self.indices_fixed = self._indices_fixed()
        self.indices_freefixed = self._indices_freefixed()

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

    @property
    def support_index(self):
        """
        A mapping from support keys to indices.
        """
        return {int(key): index for index, key in enumerate(self.nodes_fixed)}

    @property
    def nodes_free(self):
        """
        The free nodes.
        """
        return self.nodes[self.indices_free]

    @property
    def nodes_fixed(self):
        """
        The fixed nodes.
        """
        return self.nodes[self.indices_fixed]

    def _connectivity_free(self):
        """
        The connectivity matrix between edges and nodes.
        """
        return self.connectivity[:, self.indices_free]

    def _connectivity_fixed(self):
        """
        The connectivity matrix between edges and nodes.
        """
        return self.connectivity[:, self.indices_fixed]

    def _indices_free(self):
        """
        The indices of the unsupported nodes in the structure.
        """
        indices = jnp.flatnonzero(self.supports == 0, size=self.num_free)

        return indices

    def _indices_fixed(self):
        """
        The indices of the unsupported nodes in the structure.
        """
        indices = jnp.flatnonzero(self.supports, size=self.num_supports)

        return indices

    def _indices_freefixed(self):
        """
        A list with the node keys of all the nodes sorted by their node index.
        """
        # TODO: this method must be refactored to be more transparent.
        freefixed_indices = jnp.concatenate([self.indices_free,
                                             self.indices_fixed])

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
    diag_indices: jax.Array
    index_array: jax.Array
    diags: jax.Array

    def __init__(self, nodes, edges, supports, **kwargs):
        super().__init__(nodes=nodes,
                         edges=edges,
                         supports=supports,
                         **kwargs)

        # Do some precomputation to be able to construct
        # the lhs matrix through indexing
        c_free_csc = self.connectivity_scipy[:, self.indices_free]
        index_array = self._get_sparse_index_array(c_free_csc)
        self.index_array = index_array

        # Indices of data corresponding to diagonal.
        # With this array we can just index directly into the
        # CSC.data array to refer to the diagonal entries.
        self.diag_indices = self._get_sparse_diag_indices(index_array)

        # Prepare the array D st when D.T @ q we get the diagonal elements of matrix
        self.diags = self._get_sparse_diag_data(c_free_csc)

    def _connectivity_free(self):
        """
        The connectivity matrix between edges and nodes.
        """
        return BCOO.from_scipy_sparse(self.connectivity_scipy[:, self.indices_free])

    def _connectivity_fixed(self):
        """
        The connectivity matrix between edges and nodes.
        """
        return BCOO.from_scipy_sparse(self.connectivity_scipy[:, self.indices_fixed])

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
# Mesh structures
# ==========================================================================

class EquilibriumMeshStructure(EquilibriumStructure, Mesh):
    """
    An equilibrium mesh structure.
    """
    def __init__(self, vertices, faces, edges, supports, **kwargs):
        super().__init__(nodes=vertices,
                         edges=edges,
                         supports=supports,
                         vertices=vertices,
                         faces=faces,
                         **kwargs)

    @property
    def num_free(self):
        """
        The number of supports.
        """
        return self.num_vertices - self.num_supports

    @classmethod
    def from_mesh(cls, mesh):
        """
        Create a structure from a force density mesh.
        """
        vertices = list(mesh.vertices())
        edges = list(mesh.edges())

        supports = []
        for vertex in vertices:
            flag = 0.0
            if mesh.is_vertex_support(vertex):
                flag = 1.0
            supports.append(flag)

        faces = [mesh.face_vertices(fkey) for fkey in mesh.faces()]
        max_length_face = max(len(face) for face in faces)
        assert max_length_face > 2, "The mesh faces must have at least 3 vertices each"

        padded_faces = []
        for face in faces:
            len_face = len(face)
            if len_face < max_length_face:
                pad_value = face[0]
                face_padding = [pad_value] * (max_length_face - len_face)
                face = face + face_padding
            padded_faces.append(face)

        faces = np.asarray(padded_faces, dtype=DTYPE_INT_NP)
        vertices = np.asarray(vertices, dtype=DTYPE_INT_NP)
        edges = np.asarray(edges, dtype=DTYPE_INT_NP)
        supports = np.asarray(supports, dtype=DTYPE_INT_NP)

        return cls(vertices, faces, edges, supports)

    @property
    def support_index(self):
        """
        A mapping from support vertices keys to indices.
        """
        return {int(key): index for index, key in enumerate(self.vertices_fixed)}

    @property
    def vertices_free(self):
        """
        The free vertices.
        """
        return self.vertices[self.indices_free]

    @property
    def vertices_fixed(self):
        """
        The fixed vertices.
        """
        return self.vertices[self.indices_fixed]


class EquilibriumMeshStructureSparse(EquilibriumMeshStructure, EquilibriumStructureSparse, MeshSparse):
    """
    An equilibrium mesh structure.
    """
    def __init__(self, vertices, faces, edges, supports, **kwargs):
        super().__init__(vertices=vertices,
                         faces=faces,
                         edges=edges,
                         supports=supports,
                         **kwargs)
