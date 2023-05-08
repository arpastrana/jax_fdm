import numpy as np

import jax.numpy as jnp

from compas.datastructures import network_find_cycles

from compas.numerical import connectivity_matrix
from compas.numerical import face_matrix

from jax.experimental.sparse import BCOO
from jax.experimental.sparse import CSC


# ==========================================================================
# Structure
# ==========================================================================

class EquilibriumStructure:
    """
    An equilibrium structure.
    """
    def __init__(self, network):
        self._network = network

        self._connectivity = None
        self._connectivity_free = None
        self._connectivity_fixed = None
        self._connectivity_faces = None

        self._connectivity_scipy = None

        self._edges = None
        self._nodes = None
        self._faces = None

        self._num_nodes = None
        self._num_edges = None
        self._num_faces = None

        self._free_nodes = None
        self._fixed_nodes = None
        self._freefixed_nodes = None

        self._node_index = None
        self._edge_index = None
        self._anchor_index = None
        self._face_node_index = None

    @classmethod
    def from_network(cls, network):
        """
        Create a structure from a network.
        """
        return cls(network)

    @property
    def network(self):
        """
        A COMPAS network.
        """
        return self._network

    @property
    def edges(self):
        """
        A list with the edge keys of the structure.
        """
        if not self._edges:
            self._edges = list(self.network.edges())
        return self._edges

    @property
    def nodes(self):
        """
        A list with the node keys of the structure.
        """
        if not self._nodes:
            self._nodes = list(self.network.nodes())
        return self._nodes

    @property
    def faces(self):
        """
        A list with the face keys of the structure.
        """
        if not self._faces:
            self._faces = [cycle[:-1] for cycle in network_find_cycles(self.network)[1:]]
        return self._faces

    @property
    def num_edges(self):
        """
        A list with the edge keys of the structure.
        """
        if not self._num_edges:
            self._num_edges = len(self.edges)
        return self._num_edges

    @property
    def num_nodes(self):
        """
        A list with the node keys of the structure.
        """
        if not self._num_nodes:
            self._num_nodes = len(self.nodes)
        return self._num_nodes

    @property
    def num_faces(self):
        """
        A list with the face keys of the structure.
        """
        if not self._num_faces:
            self._num_faces = len(self.faces)
        return self._num_faces

    @property
    def face_node_index(self):
        """
        A list with the face keys of the structure.
        """
        if not self._face_node_index:
            face_index = []
            for face in self.faces:
                face_index.append([self.node_index[node] for node in face])
            self._face_node_index = face_index
        return self._face_node_index

    @property
    def node_index(self):
        """
        A dictionary between node keys and their enumeration indices.
        """
        if not self._node_index:
            self._node_index = self.network.key_index()
        return self._node_index

    @property
    def anchor_index(self):
        """
        A dictionary between node anchor keys and their enumeration indices.
        """
        if not self._anchor_index:
            self._anchor_index = {key: index for index, key in enumerate(self.network.nodes_anchors())}
        return self._anchor_index

    @property
    def edge_index(self):
        """
        A dictionary between edge keys and their enumeration indices.
        """
        if not self._edge_index:
            self._edge_index = self.network.uv_index()
        return self._edge_index

    @property
    def connectivity(self):
        """
        The connectivity of the network encoded as an ncidence matrix.
        """
        if self._connectivity is None:
            node_idx = self.node_index
            edges = [(node_idx[u], node_idx[v]) for u, v in self.network.edges()]

            # NOTE: Dense array
            # Currently there is a JAX bug that prevents us from using the sparse format with the connectivity matrix.
            # When `todense()` is removed from the next line, we get the following error:
            # TypeError: Value Zero(ShapedArray(float64[193,3])) with type <class 'jax._src.ad_util.Zero'> is not a valid JAX type

            con = connectivity_matrix(edges, "array")
            self._connectivity = jnp.asarray(con)

        return self._connectivity

    @property
    def connectivity_scipy(self):
        """
        The connectivity of the network encoded as an incidence matrix in CSC format.
        """
        if self._connectivity_scipy is None:
            node_idx = self.node_index
            edges = [(node_idx[u], node_idx[v]) for u, v in self.network.edges()]
            # We should get a CSC representation since we are interested in slicing columns
            self._connectivity_scipy = connectivity_matrix(edges, "csc")

        return self._connectivity_scipy

    @property
    def connectivity_fixed(self):
        """
        The connectivity of the fixed nodes of the network.
        """
        if self._connectivity_fixed is None:
            self._connectivity_fixed = self.connectivity[:, self.fixed_nodes]

        return self._connectivity_fixed

    @property
    def connectivity_free(self):
        """
        The connectivity of the free nodes of the network.
        """
        if self._connectivity_free is None:
            self._connectivity_free = self.connectivity[:, self.free_nodes]

        return self._connectivity_free

    @property
    def connectivity_faces(self):
        """
        The connectivity of the face cycles of a network encoded as as matrix.
        """
        if self._connectivity_faces is None:
            self._connectivity_faces = face_matrix(self.face_node_index, rtype="array")
        return self._connectivity_faces

    @property
    def free_nodes(self):
        """
        Returns a list with the indices of the anchored nodes.
        """
        if not self._free_nodes:
            self._free_nodes = [self.node_index[node] for node in self.network.nodes_free()]
        return self._free_nodes

    @property
    def fixed_nodes(self):
        """
        Returns a list with the indices of the anchored nodes.
        """
        if not self._fixed_nodes:
            self._fixed_nodes = [self.node_index[node] for node in self.network.nodes_fixed()]
        return self._fixed_nodes

    @property
    def freefixed_nodes(self):
        """
        A list with the node keys of all the nodes sorted by their node index.
        """
        # TODO: this method must be refactored to be more transparent.
        if not self._freefixed_nodes:
            freefixed_nodes = self.free_nodes + self.fixed_nodes
            indices = {node: index for index, node in enumerate(freefixed_nodes)}
            sorted_indices = []
            for _, index in sorted(indices.items(), key=lambda item: item[0]):
                sorted_indices.append(index)
            self._freefixed_nodes = tuple(sorted_indices)

        return self._freefixed_nodes


# ==========================================================================
# Structure
# ==========================================================================

class EquilibriumStructureSparse(EquilibriumStructure):
    """
    An equilibrium structure.
    """
    def __init__(self, network):
        super().__init__(network)

        # Do some precomputation to be able to construct the lhs matrix through indexing
        c_free_csc = self.connectivity_scipy[:, self.free_nodes]
        index_array = self._get_sparse_index_array(c_free_csc)
        self.index_array = index_array

        # Indices of data corresponding to diagonal.
        # With this array we can just index directly into the CSC.data array to refer to the diagonal entries.
        self.diag_indices = self._get_sparse_diag_indices(index_array)

        # Prepare the array D st when D.T @ q we get the diagonal elements of matrix.
        self.diags = self._get_sparse_diag_data(c_free_csc)

        self.init()

    def init(self):
        """
        Warm start properties.

        Otherwise we get a leakage error:

        jax._src.errors.UnexpectedTracerError: Encountered an unexpected tracer.
        A function transformed by JAX had a side effect, allowing for a reference to an
        intermediate value with type float64[611] wrapped in a DynamicJaxprTracer
        to escape the scope of the transformation.
        """
        self.connectivity
        self.connectivity_free
        self.connectivity_fixed
        self.free_nodes
        self.fixed_nodes
        self.freefixed_nodes

    @property
    def connectivity_fixed(self):
        """
        The connectivity of the fixed nodes of the network.
        """
        if self._connectivity_fixed is None:
            con_fixed = BCOO.from_scipy_sparse(self.connectivity_scipy[:, self.fixed_nodes])
            self._connectivity_fixed = con_fixed

        return self._connectivity_fixed

    @property
    def connectivity_free(self):
        """
        The connectivity of the free nodes of the network.
        """
        if self._connectivity_free is None:
            con_free = BCOO.from_scipy_sparse(self.connectivity_scipy[:, self.free_nodes])
            self._connectivity_free = con_free

        return self._connectivity_free

    @staticmethod
    def _get_sparse_index_array(c_free_csc):
        """
        Create an index array such that the off-diagonals can index into the force density vector.

        This array is used to create the off-diagonal entries of the lhs matrix.
        """
        force_density_modified_c_free_csc = c_free_csc.copy()
        force_density_modified_c_free_csc.data *= np.take(np.arange(c_free_csc.shape[0]) + 1, c_free_csc.indices)
        index_array = -(c_free_csc.T @ force_density_modified_c_free_csc)

        # The diagonal entries should be set to 0 so that it indexes
        # into a valid entry, but will later be overwritten.
        index_array.setdiag(0)

        return index_array.astype(int)

    @staticmethod
    def _get_sparse_diag_data(c_free_csc):
        """
        The diagonal of the lhs matrix is the sum of force densities for
        each outgoing/incoming edge on the node.

        We create the `diags` matrix such that when we multiply it with the
        force density vector we get the diagonal.
        """
        diags_data = jnp.ones_like(c_free_csc.data)

        return CSC((diags_data, c_free_csc.indices, c_free_csc.indptr), shape=c_free_csc.shape)

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
