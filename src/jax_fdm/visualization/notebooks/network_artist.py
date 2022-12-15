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

from jax_fdm.visualization.artists import FDNetworkArtist

from jax_fdm.visualization.notebooks import Arrow


__all__ = ["FDNetworkNotebookArtist"]


class FDNetworkNotebookArtist(FDNetworkArtist):
    """
    An artist that draws a force density network to a viewer.
    """
    default_opacity = 0.75
    arrow_bodywidth = 0.012
    arrow_headportion = 0.12
    arrow_headwidth = 0.04

    def __init__(self, network, viewer, *args, **kwargs):
        super().__init__(network, *args, **kwargs)
        # NOTE: this attribute has to be handled by a ViewerArtist
        self.viewer = viewer

        self.viewer_edges = {}
        self.viewer_nodes = {}
        self.viewer_loads = {}
        self.viewer_reactions = {}

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def objects(self):
        """
        Yield the next viewer object.
        """
        objects = {}

        objects.update(self.viewer_edges)
        objects.update(self.viewer_nodes)
        objects.update(self.viewer_loads)
        objects.update(self.viewer_reactions)

        for obj in objects.values():
            yield obj

    # ==========================================================================
    # Add
    # ==========================================================================

    def add(self):
        """
        Add the elements of the network to the viewer.
        """
        if self.collection_nodes:
            self.viewer_nodes = self.add_nodes()
        if self.collection_edges:
            self.viewer_edges = self.add_edges()
        if self.collection_reactions:
            self.viewer_reactions = self.add_reactions()
        if self.collection_loads:
            self.viewer_loads = self.add_loads()

    # ==========================================================================
    # Edges
    # ==========================================================================

    def draw_edge(self, edge, width, *args, **kwargs):
        """
        Draw an edge.
        """
        u, v = edge
        network = self.network

        plane = Plane(network.edge_midpoint(u, v), network.edge_direction(u, v))
        circle = Circle(plane, width / 2.0)

        return Cylinder(circle, height=network.edge_length(u, v))

    def add_edge(self, edge, color):
        """
        Add an edge to the viewer.
        """
        return self.viewer.add(edge, facecolor=color, linecolor=color)

    def add_edges(self):
        """
        Add the edges of the network to the viewer.
        """
        edges = {}

        for edge, cylinder in self.collection_edges.items():
            color = self.edge_color[edge]
            obj = self.add_edge(cylinder, color)
            edges[edge] = obj

        return edges

    # ==========================================================================
    # Draw one element
    # ==========================================================================

    def draw_node(self, node, size, *args, **kwargs):
        """
        Draw a node.
        """
        return Sphere(self.network.node_coordinates(node), radius=size/2.0)

    def add_node(self, node, color):
        """
        Add one node to the viewer.
        """
        return self.viewer.add(node, facecolor=color, linecolor=color)

    def add_nodes(self):
        """
        Add the nodes to viewer.
        """
        nodes = {}

        for node, sphere in self.collection_nodes.items():
            color = self.node_color[node]
            obj = self.add_node(sphere, color)
            nodes[node] = obj

        return nodes

    # ==========================================================================
    # Loads
    # ==========================================================================

    def draw_load(self, node, scale, *args, **kwargs):
        """
        Draw a load vector at a node.
        """
        vector = self.network.node_load(node)

        if length_vector(vector) < self.load_tol:
            return

        xyz = self.network.node_coordinates(node)
        # shift start to make arrow head touch the node the load is applied to
        start = add_vectors(xyz, scale_vector(vector, -scale))
        # shift start to account for half size of edge thickness
        widths = [self.edge_width[edge] for edge in self.network.connected_edges(node)]
        start = add_vectors(start, scale_vector(vector, -max(widths)))

        return self.draw_vector(vector, start, scale)

    def add_load(self, load, color):
        """
        Add one load to the viewer.
        """
        return self.viewer.add(load, facecolor=color, linecolor=color)

    def add_loads(self):
        """
        Add the loads to viewer.
        """
        loads = {}

        for node, arrow in self.collection_loads.items():
            obj = self.add_load(arrow, self.default_loadcolor)
            loads[node] = obj

        return loads

    # ==========================================================================
    # Reaction forces
    # ==========================================================================

    def draw_reaction(self, node, scale, *args, **kwargs):
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

    def add_reaction(self, reaction, color):
        """
        Add one reaction force to the viewer.
        """
        return self.viewer.add(reaction, facecolor=color, linecolor=color)

    def add_reactions(self):
        """
        Add the reaction forces to viewer.
        """
        reactions = {}

        for node, arrow in self.collection_reactions.items():
            obj = self.add_reaction(arrow, self.default_reactioncolor)
            reactions[node] = obj

        return reactions

    # ==========================================================================
    # Helpers
    # ==========================================================================

    def draw_vector(self, vector, start, scale):
        """
        Draw a vector.
        """
        vector_scaled = scale_vector(vector, scale)
        end = add_vectors(start, vector_scaled)

        return Arrow(Point(*start),
                     Vector.from_start_end(start, end),
                     head_portion=self.arrow_headportion,
                     head_width=self.arrow_headwidth,
                     body_width=self.arrow_bodywidth)
