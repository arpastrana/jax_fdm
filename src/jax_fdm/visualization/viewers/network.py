from math import fabs

from compas.geometry import Plane
from compas.geometry import Circle
from compas.geometry import Sphere
from compas.geometry import Cylinder
from compas.geometry import Point
from compas.geometry import Vector
from compas.geometry import scale_vector
from compas.geometry import add_vectors
from compas.geometry import length_vector

from compas_view2.shapes import Arrow
from compas_view2.collections import Collection

from jax_fdm.visualization.artists import FDNetworkArtist


__all__ = ["FDNetworkViewerArtist"]


class FDNetworkViewerArtist(FDNetworkArtist):
    """
    An artist that draws a force density network to a viewer.
    """
    default_opacity = 0.75
    arrow_headportion = 0.2
    arrow_headwidth = 0.07
    arrow_bodywidth = 0.02

    def __init__(self, network, viewer, *args, **kwargs):
        super().__init__(network, *args, **kwargs)
        # NOTE: this attribute gotta be handled by a ViewerArtist
        self.viewer = viewer

    def draw_node(self, node, size, color):
        """
        Draw a node.
        """
        sphere = Sphere(self.network.node_coordinates(node), radius=size/2.0)

        return self.viewer.add(sphere,
                               facecolor=color,
                               linecolor=color,
                               show_edges=True,
                               opacity=self.default_opacity)

    def draw_edge(self, edge, width, color):
        """
        Draw an edge.
        """
        u, v = edge
        network = self.network

        plane = Plane(network.edge_midpoint(u, v), network.edge_direction(u, v))
        circle = Circle(plane, width / 2.0)
        cylinder = Cylinder(circle, height=network.edge_length(u, v))

        return self.viewer.add(cylinder,
                               facecolor=color,
                               linecolor=color,
                               show_edges=True,
                               opacity=self.default_opacity)

    def draw_loads(self):
        """
        Draw the loads at the nodes of the network as a collection.
        """
        loads = []
        for node in self.nodes:
            load = self.draw_load(node, self.load_scale)
            if load:
                loads.append(load)

        return [self.viewer.add(Collection(loads),
                                facecolor=self.load_color,
                                linecolor=self.load_color,
                                show_edges=True,
                                opacity=self.default_opacity)]

    def draw_load(self, node, scale):
        """
        Draw a load vector at a node.
        """
        vector = self.network.node_load(node)
        start = self.network.node_coordinates(node)

        if length_vector(vector) < self.load_tol:
            return

        return self.draw_vector(vector, start, scale)

    def draw_reactions(self):
        """
        Draw the reactions at the nodes of the network.
        """
        reactions = []

        for node in self.nodes:
            reaction = self.draw_reaction(node, self.reaction_scale)
            if reaction:
                reactions.append(reaction)

        return [self.viewer.add(Collection(reactions),
                                facecolor=self.reaction_color,
                                linecolor=self.reaction_color,
                                show_edges=True,
                                opacity=self.default_opacity)]

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
        return self.draw_vector(scale_vector(vector, -1.0), start, scale)

    @staticmethod
    def draw_vector(vector, start, scale):
        """
        Draw a vector.
        """
        vector_scaled = scale_vector(vector, scale)
        end = add_vectors(start, vector_scaled)

        return Arrow(Point(*start),
                     Vector.from_start_end(start, end),
                     head_portion=0.2,
                     head_width=0.07,
                     body_width=0.02)


if __name__ == "__main__":
    import os

    from compas.colors import Color

    from jax_fdm import DATA

    from jax_fdm.datastructures import FDNetwork
    from jax_fdm.equilibrium import fdm
    from jax_fdm.visualization import Viewer

    network = FDNetwork.from_json(os.path.join(DATA, "json/arch.json"))
    network.edges_forcedensities(q=-5.0)
    network.nodes_supports(keys=[node for node in network.nodes() if network.is_leaf(node)])
    network.nodes_loads([0.0, 0.0, -0.2], keys=network.nodes_free())

    network = fdm(network)
    viewer = Viewer()
    viewer.add(network, edgewidth=0.1, loadscale=2.0, reactionscale=0.5, reactioncolor=Color.pink())
    viewer.show()
