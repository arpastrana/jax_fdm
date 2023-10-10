"""
A force density network.
"""

from compas.datastructures import Network

from jax_fdm.datastructures import FDDatastructure


class FDNetwork(Network, FDDatastructure):
    """
    A force density network.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.update_default_edge_attributes({"q": 0.0,
                                             "length": 0.0,
                                             "force": 0.0,
                                             "px": 0.0,
                                             "py": 0.0,
                                             "pz": 0.0})

        self.update_default_node_attributes({"x": 0.0,
                                             "y": 0.0,
                                             "z": 0.0,
                                             "px": 0.0,
                                             "py": 0.0,
                                             "pz": 0.0,
                                             "rx": 0.0,
                                             "ry": 0.0,
                                             "rz": 0.0,
                                             "is_support": False})

    # ----------------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------------

    @classmethod
    def from_mesh(cls, mesh):
        """
        Create a force density network from a mesh.
        """
        nodes = {vkey: mesh.vertex_coordinates(vkey) for vkey in mesh.vertices()}
        network = cls.from_nodes_and_edges(nodes, mesh.edges())

        for node in network.nodes():
            attrs = mesh.vertex_attributes(node)
            network.node_attributes(node, names=attrs.keys(), values=attrs.values())

        for edge in network.edges():
            attrs = mesh.edge_attributes(edge)
            network.edge_attributes(edge, names=attrs.keys(), values=attrs.values())

        return network

    # ----------------------------------------------------------------------
    # Nodes
    # ----------------------------------------------------------------------

    def nodes_coordinates(self, keys=None, axes="xyz"):
        """
        Gets or sets the x, y, z coordinates of a list of nodes.
        """
        keys = keys or self.nodes()
        return [self.node_coordinates(node, axes) for node in keys]

    def nodes_fixedcoordinates(self, keys=None, axes="xyz"):
        """
        Gets the x, y, z coordinates of the anchors of the network.
        """
        if keys:
            keys = {key for key in keys if self.is_node_support(key)}
        else:
            keys = self.nodes_fixed()

        return [self.node_coordinates(node, axes) for node in keys]

    def number_of_anchors(self):
        """
        The number of anchored nodes.
        """
        return len(list(self.nodes_anchors()))

    def number_of_supports(self):
        """
        The number of supported nodes.
        """
        return len(list(self.nodes_supports()))

    def node_support(self, key):
        """
        Sets a node as a fixed anchor.
        """
        return self.node_attribute(key=key, name="is_support", value=True)

    def node_anchor(self, key):
        """
        Sets a node as a fixed anchor.
        """
        return self.node_support(key)

    def is_node_support(self, key):
        """
        Test if the node is a fixed node.
        """
        return self.node_attribute(key=key, name="is_support")

    def nodes_supports(self, keys=None):
        """
        Gets or sets the node keys where a support has been assigned.
        """
        if keys is None:
            return self.nodes_where({"is_support": True})

        return self.nodes_attribute(name="is_support", value=True, keys=keys)

    def nodes_fixed(self, keys=None):
        """
        Gets or sets the node keys where a support has been assigned.
        """
        return self.nodes_supports(keys)

    def nodes_anchors(self, keys=None):
        """
        Gets or sets the node keys where an anchor has been assigned.
        """
        return self.nodes_supports(keys)

    def nodes_free(self):
        """
        The keys of the nodes where there is no support assigned.
        """
        return self.nodes_where({"is_support": False})

    def node_load(self, key, load=None):
        """
        Gets or sets a load to the nodes of the network.
        """
        return self.node_attributes(key=key, names=("px", "py", "pz"), values=load)

    def nodes_loads(self, load=None, keys=None):
        """
        Gets or sets a load to the nodes of the network.
        """
        return self.nodes_attributes(names=("px", "py", "pz"), values=load, keys=keys)

    def nodes_residual(self, keys=None):
        """
        Gets the residual forces of the nodes of the network.
        """
        return self.nodes_attributes(names=("rx", "ry", "rz"), keys=keys)

    def node_residual(self, key):
        """
        Gets the residual force of a single node of the network.
        """
        return self.node_attributes(key=key, names=("rx", "ry", "rz"))

    def nodes_reactions(self, keys=None):
        """
        Gets the reaction forces of the nodes of the network.
        """
        keys = keys or self.nodes_fixed()
        return self.nodes_attributes(names=("rx", "ry", "rz"), keys=keys)

    def node_reaction(self, key):
        """
        Gets the reaction force of a single node of the network.
        """
        return self.node_attributes(key=key, names=("rx", "ry", "rz"))

    # ----------------------------------------------------------------------
    # Edges
    # ----------------------------------------------------------------------

    def is_edge_supported(self, key):
        """
        Test if any of the two nodes connected by the edge is a support.
        """
        return any(self.is_node_support(node) for node in key)

    def is_edge_fully_supported(self, key):
        """
        Test if the two nodes connected the edge are a support.
        """
        return all(self.is_node_support(node) for node in key)

    # ----------------------------------------------------------------------
    # Datastructure properties
    # ----------------------------------------------------------------------

    def parameters(self):
        """
        Return the design parameters of the network.
        """
        q = self.edges_forcedensities()
        xyz_fixed = self.nodes_fixedcoordinates()
        loads = self.nodes_loads()

        return q, xyz_fixed, loads
