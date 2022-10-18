"""
A catalogue of force density networks.
"""
from math import fabs

from compas.datastructures import Network
from compas.geometry import transform_points


class FDNetwork(Network):
    """
    A force density network.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        self.update_default_edge_attributes({"q": 0.0,
                                             "length": 0.0,
                                             "force": 0.0})

    def nodes_coordinates(self, keys=None, axes="xyz"):
        """
        Gets or sets the x, y, z coordinates of a list of nodes.
        """
        keys = keys or self.nodes()
        return [self.node_coordinates(node, axes) for node in keys]

    def node_support(self, key):
        """
        Sets a node as a fixed anchor.
        """
        return self.node_attribute(key=key, name="is_support", value=True)

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

    def nodes_free(self):
        """
        The keys of the nodes where there is no support assigned.
        """
        return self.nodes_where({"is_support": False})

    def edge_forcedensity(self, key, q=None):
        """
        Gets or sets the force density on a single edge.
        """
        return self.edge_attribute(name="q", value=q, key=key)

    def edges_forcedensities(self, q=None, keys=None):
        """
        Gets or sets the force densities on a list of edges.
        """
        return self.edges_attribute(name="q", value=q, keys=keys)

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
        return self.nodes_attributes(names=("rx", "ry", "rz"), keys=keys)

    def node_reaction(self, key):
        """
        Gets the reaction force of a single node of the network.
        """
        return self.node_attributes(key=key, names=("rx", "ry", "rz"))

    def edge_force(self, key):
        """
        Gets the forces at a single edge the network.
        """
        return self.edge_attribute(key=key, name="force")

    def edges_forces(self, keys=None):
        """
        Gets the forces on the edges of the network.
        """
        return self.edges_attribute(keys=keys, name="force")

    def edges_lengths(self, keys=None):
        """
        Gets the lengths on the edges of the network.
        """
        return self.edges_attribute(keys=keys, name="length")

    def edge_loadpath(self, key):
        """
        Gets the load path at a single edge the network.
        """
        force = self.edge_attribute(key=key, name="force")
        length = self.edge_attribute(key=key, name="length")
        return fabs(force * length)

    def edges_loadpaths(self, keys=None):
        """
        Gets the load path on the edges of the network.
        """
        keys = keys or self.edges()
        for key in keys:
            yield self.edge_loadpath(key)

    def loadpath(self):
        """
        Gets the total load path of the network.
        """
        return sum(list(self.edges_loadpaths()))

    def print_stats(self, other_stats=None):
        """
        Print information aboud the equilibrium state of the network.
        """
        stats = {"FDs": self.edges_forcedensities(),
                 "Forces": self.edges_forces(),
                 "Lengths": self.edges_lengths()}

        other_stats = other_stats or dict()
        stats.update(other_stats)

        print("\n***Network stats***")
        print(f"Load path: {round(self.loadpath(), 3)}")

        for name, vals in stats.items():

            minv = round(min(vals), 3)
            maxv = round(max(vals), 3)
            meanv = round(sum(vals) / len(vals), 3)

            print(f"{name}\t\tMin: {minv}\tMax: {maxv}\tMean: {meanv}")

    def transformed(self, transformation):
        """
        Return a transformed copy of the network.
        """
        network = self.copy()

        attr_groups = [("x", "y", "z"),
                       ("px", "py", "pz"),
                       ("rx", "ry", "rz")]

        nodes = list(self.nodes())
        for attr_names in attr_groups:
            xyz_t = transform_points(self.nodes_attributes(names=attr_names, keys=nodes), transformation)
            for node, xyz in zip(nodes, xyz_t):
                network.node_attributes(node, names=attr_names, values=xyz)

        return network
