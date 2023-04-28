import jax.numpy as jnp

from compas.datastructures import network_find_cycles

from compas.numerical import connectivity_matrix
from compas.numerical import face_matrix

from jax.experimental.sparse import BCOO


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

        self._free_nodes = None
        self._fixed_nodes = None
        self._freefixed_nodes = None

        self._node_index = None
        self._edge_index = None
        self._anchor_index = None
        self._face_node_index = None

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
        if not self.nodes:
            self._nodes = list(self.network.nodes())

    @property
    def faces(self):
        """
        A list with the face keys of the structure.
        """
        if not self._faces:
            self._faces = [cycle[:-1] for cycle in network_find_cycles(self.network)[1:]]
        return self._faces

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

class SparseEquilibriumStructure(EquilibriumStructure):
    """
    An equilibrium structure.
    """
    def __init__(self, network):
        super().__init__(network)
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
        self.connectivity_free
        self.connectivity_fixed

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
