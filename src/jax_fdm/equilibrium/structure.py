import numpy as np

from compas.numerical import connectivity_matrix


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

        self._edges = None
        self._nodes = None

        self._free_nodes = None
        self._fixed_nodes = None
        self._freefixed_nodes = None

        self._node_index = None
        self._edge_index = None
        self._anchor_index = None

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
        The connectivity of the network encoded as a branch-node list of lists.
        """
        if self._connectivity is None:
            node_idx = self.node_index
            edges = [(node_idx[u], node_idx[v]) for u, v in self.network.edges()]
            self._connectivity = np.array(connectivity_matrix(edges, "list"), dtype=np.float64)
        return self._connectivity

    @property
    def connectivity_fixed(self):
        """
        The connectivity of the fixed nodes of the network.
        """
        if self._connectivity_fixed is None:
            fixed_nodes = self.fixed_nodes
            self._connectivity_fixed = self.connectivity[:, fixed_nodes]
        return self._connectivity_fixed

    @property
    def connectivity_free(self):
        """
        The connectivity of the free nodes of the network.
        """
        if self._connectivity_free is None:
            free_nodes = self.free_nodes
            self._connectivity_free = self.connectivity[:, free_nodes]
        return self._connectivity_free

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
        TODO: this method must be refactored to be more transparent.
        """
        if not self._freefixed_nodes:
            freefixed_nodes = self.free_nodes + self._fixed_nodes
            indices = {node: index for index, node in enumerate(freefixed_nodes)}
            sorted_indices = []
            for _, index in sorted(indices.items(), key=lambda item: item[0]):
                sorted_indices.append(index)
            self._freefixed_nodes = tuple(sorted_indices)

        return self._freefixed_nodes
