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

try:
    from compas_view2.shapes import Arrow
except ImportError:
    pass

from jax_fdm.visualization.artists import FDNetworkArtist


__all__ = ["FDNetworkViewerArtist"]


class FDNetworkViewerArtist(FDNetworkArtist):
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
    # Update
    # ==========================================================================

    def update(self):
        """
        Update the elements of the network drawn by this artist.
        """
        if self.viewer_nodes:
            self.update_nodes()
        if self.viewer_edges:
            self.update_edges()
        if self.viewer_reactions:
            self.update_reactions()
        if self.viewer_loads:
            self.update_loads()

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
        return self.viewer.add(edge,
                               facecolor=color,
                               linecolor=color,
                               show_edges=True,
                               opacity=self.default_opacity)

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

    def update_edge(self, edge, width, color):
        """
        Update an edge.
        """
        u, v = edge
        network = self.network
        cylinder = self.collection_edges[edge]

        plane = Plane(network.edge_midpoint(u, v), network.edge_direction(u, v))
        circle = Circle(plane, width / 2)

        cylinder.circle = circle
        cylinder.height = self.network.edge_length(u, v)

        obj = self.viewer_edges[edge]
        obj.linecolor = color
        obj.facecolor = color

    def update_edges(self):
        """
        Update the edges of the network.
        """
        self.edge_color = self._init_edgecolor
        self.edge_width = self._init_edgewidth

        for edge in self.edges:
            width = self.edge_width[edge]
            color = self.edge_color[edge]
            self.update_edge(edge, width, color)

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
        return self.viewer.add(node,
                               facecolor=color,
                               linecolor=color,
                               show_edges=False,
                               opacity=self.default_opacity)

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

    def update_node(self, node, size):
        """
        Update a node in the viewer.
        """
        sphere = self.collection_nodes[node]
        sphere.point = self.network.node_coordinates(node)
        sphere.radius = size / 2.0

    def update_nodes(self):
        """
        Update the nodes in the viewer.
        """
        for node in self.collection_nodes.keys():
            size = self.node_size[node]
            self.update_node(node, size)

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
        return self.viewer.add(load,
                               facecolor=color,
                               linecolor=color,
                               show_edges=True,
                               opacity=self.default_opacity)

    def add_loads(self):
        """
        Add the loads to viewer.
        """
        loads = {}

        for node, arrow in self.collection_loads.items():
            obj = self.add_load(arrow, self.default_loadcolor)
            loads[node] = obj

        return loads

    def update_load(self, node, scale):
        """
        Update a load in the viewer.
        """
        arrow_new = self.draw_load(node, scale)

        arrow = self.collection_loads[node]
        arrow.position = arrow_new.position
        arrow.direction = arrow_new.direction

    def update_loads(self):
        """
        Update the loads in the viewer.
        """
        for node in self.collection_loads.keys():
            self.update_load(node, self.load_scale)

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
        return self.viewer.add(reaction,
                               facecolor=color,
                               linecolor=color,
                               show_edges=True,
                               opacity=self.default_opacity)

    def add_reactions(self):
        """
        Add the reaction forces to viewer.
        """
        reactions = {}

        for node, arrow in self.collection_reactions.items():
            obj = self.add_reaction(arrow, self.default_reactioncolor)
            reactions[node] = obj

        return reactions

    def update_reaction(self, node, scale):
        """
        Update a reaction force in the viewer.
        """
        arrow_new = self.draw_reaction(node, scale)

        arrow = self.collection_reactions[node]
        arrow.position = arrow_new.position
        arrow.direction = arrow_new.direction

    def update_reactions(self):
        """
        Update the reaction forces in the viewer.
        """
        for node in self.collection_reactions.keys():
            self.update_reaction(node, self.reaction_scale)

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
