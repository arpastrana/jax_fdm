from math import fabs

from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.geometry import scale_vector
from compas.geometry import Point
from compas.geometry import Vector

from compas_plotters.artists import NetworkArtist

from jax_fdm.visualization.artists import FDNetworkArtist


class FDNetworkPlotterArtist(FDNetworkArtist, NetworkArtist):
    """
    An artist that knows how to draw a force density network in a plotter.
    """
    def draw_nodes(self):
        """
        Draw the nodes of the network.
        """
        NetworkArtist.draw_nodes(self)
        return []

    def draw_edges(self):
        """
        Draw the edges of the network.
        """
        NetworkArtist.draw_edges(self)
        return []

    def draw_loads(self):
        """
        Draw the loads at the nodes of the network.
        """
        loads = []
        for node in self.nodes:
            load = self.draw_load(node, self.load_scale)
            if load:
                loads.append(load)

        return loads

    def draw_reactions(self):
        """
        Draw the reactions at the nodes of the network.
        """
        reactions = []
        for node in self.nodes:
            reaction = self.draw_reaction(node, self.reaction_scale)
            if reaction:
                reactions.append(reaction)

        return reactions

    def draw_reaction(self, node, scale):
        """
        Draw a reaction vector at a node.
        """
        network = self.network

        vector = network.node_reaction(node)
        start = network.node_coordinates(node)

        if length_vector(vector) < self.reaction_tol:
            return

        # shift starting point if max force of connected edges is compressive
        forces = [network.edge_force(e) for e in network.connected_edges(node)]
        max_force = max(forces, key=lambda f: fabs(f))
        if max_force < 0.0:
            start = add_vectors(start, scale_vector(vector, scale))

        # reverse vector to display direction of reaction forces
        reaction = self.draw_vector(scale_vector(vector, -1.0), start, scale)

        return self.plotter.add(reaction, point=Point(*start), color=self.reaction_color)

    def draw_load(self, node, scale):
        """
        Draw a load vector at a node.
        """
        vector = self.network.node_load(node)
        start = self.network.node_coordinates(node)

        if length_vector(vector) < self.load_tol:
            return

        load = self.draw_vector(vector, start, scale)

        return self.plotter.add(load, point=Point(*start), color=self.load_color)

    @staticmethod
    def draw_vector(vector, start, scale):
        """
        Draw a vector as an arrow.
        """
        vector_scaled = scale_vector(vector, scale)
        end = add_vectors(start, vector_scaled)

        return Vector.from_start_end(start, end)
