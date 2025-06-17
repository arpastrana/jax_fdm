"""
A force density mesh.
"""
from math import fabs
from statistics import stdev

from compas.datastructures import Datastructure


class FDDatastructure(Datastructure):
    """
    A force density datastructure.
    """
    # ----------------------------------------------------------------------
    # Edges
    # ----------------------------------------------------------------------

    def edge_load(self, key, load=None):
        """
        Gets or sets a load on an edge.
        """
        return self.edge_attributes(key, names=("px", "py", "pz"), values=load)

    def edge_forcedensity(self, key, q=None):
        """
        Gets or sets the force density on a single edge.
        """
        return self.edge_attribute(key, name="q", value=q)

    def edge_force(self, key):
        """
        Gets the forces at a single edge the network.
        """
        return self.edge_attribute(key, name="force")

    def edge_loadpath(self, key):
        """
        Gets the load path at a single edge the network.
        """
        force = self.edge_attribute(key, name="force")
        length = self.edge_attribute(key, name="length")
        return fabs(force * length)

    def edges_forcedensities(self, q=None, keys=None):
        """
        Gets or sets the force densities on a list of edges.
        """
        return self.edges_attribute(name="q", value=q, keys=keys)

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

    def edges_loads(self, load=None, keys=None):
        """
        Gets or sets a load to the edges of the datastructure.
        """
        return self.edges_attributes(names=("px", "py", "pz"), values=load, keys=keys)

    def edges_loadpaths(self, keys=None):
        """
        Gets the load path on the edges of the network.
        """
        keys = keys or self.edges()
        for key in keys:
            yield self.edge_loadpath(key)

    # ----------------------------------------------------------------------
    # Datastructure properties
    # ----------------------------------------------------------------------

    def loadpath(self):
        """
        Gets the total load path of the network.
        """
        return sum(list(self.edges_loadpaths()))

    def print_stats(self, other_stats=None, ndigits=3):
        """
        Print information aboud the equilibrium state of the network.
        """
        edges_pos = []
        edges_neg = []
        for edge in self.edges():
            _edges = edges_neg
            if self.edge_forcedensity(edge) > 0.0:
                _edges = edges_pos
            _edges.append(edge)

        stats = {"FDs [-]": self.edges_forcedensities(keys=edges_neg),
                 "FDs [+]": self.edges_forcedensities(keys=edges_pos),
                 "Forces [-]": self.edges_forces(keys=edges_neg),
                 "Forces [+]": self.edges_forces(keys=edges_pos),
                 "Lengths": self.edges_lengths()}

        other_stats = other_stats or {}
        stats.update(other_stats)

        print(f"\n***{self.__class__.__name__} stats***")
        print(f"Load path: {round(self.loadpath(), ndigits)}")

        for name, vals in stats.items():

            if not vals:
                continue

            minv = round(min(vals), ndigits)
            maxv = round(max(vals), ndigits)
            meanv = round(sum(vals) / len(vals), ndigits)
            stdv = round(stdev(vals), ndigits)
            name = "{:<18}".format(name)

            print(f"{name}\tMin: {minv}\tMax: {maxv}\tMean: {meanv}\tStDev: {stdv}")
