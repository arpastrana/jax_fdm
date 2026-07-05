from math import fabs

from compas.datastructures import Mesh
from compas.geometry import Cylinder
from compas.geometry import Line
from compas.geometry import Sphere
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.geometry import normalize_vector
from compas.geometry import scale_vector
from jax_fdm.visualization.artists import FDNetworkArtist
from jax_fdm.visualization.shapes import Arrow

__all__ = ["FDNetworkViewerArtist"]


class FDNetworkViewerArtist(FDNetworkArtist):
    """
    An artist that draws a force density network to a :class:`compas_viewer.Viewer`.

    The artist builds plain COMPAS geometry (spheres for nodes, cylinders for
    edges, arrow meshes for load and reaction vectors) and pushes it into the
    viewer scene. It keeps a handle on every scene object so that an animation
    loop can mutate the geometry in place and re-read it with a single
    ``scene_object.update(update_data=True)``.
    """
    default_opacity = 0.75
    arrow_bodywidth = 0.012
    arrow_headportion = 0.12
    arrow_headwidth = 0.04

    def __init__(self, network, viewer, *args, **kwargs):
        super().__init__(network, *args, **kwargs)
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
        Yield the scene objects drawn by this artist.
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
        Add the elements of the network to the viewer scene.
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
        Update the elements of the network drawn by this artist in place.
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
        Draw an edge as a cylinder.
        """
        start, end = self.network.edge_coordinates(edge)
        line = Line(start, end)

        return Cylinder.from_line_and_radius(line, width / 2.0)

    def add_edge(self, cylinder, color):
        """
        Add one edge to the viewer scene.
        """
        return self.viewer.scene.add(cylinder,
                                     facecolor=color,
                                     linecolor=color,
                                     opacity=self.default_opacity)

    def add_edges(self):
        """
        Add the edges of the network to the viewer scene.
        """
        edges = {}

        for edge, cylinder in self.collection_edges.items():
            color = self.edge_color[edge]
            edges[edge] = self.add_edge(cylinder, color)

        return edges

    def update_edge(self, edge, width, color):
        """
        Update an edge in place.
        """
        cylinder = self.collection_edges[edge]

        start, end = self.network.edge_coordinates(edge)
        line = Line(start, end)
        # Mutate the stored geometry in place so the scene object re-reads it.
        cylinder.frame = Cylinder.from_line_and_radius(line, width / 2.0).frame
        cylinder.height = line.length
        cylinder.radius = width / 2.0

        obj = self.viewer_edges[edge]
        obj.facecolor = color
        obj.linecolor = color
        obj.update(update_data=True)

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
    # Nodes
    # ==========================================================================

    def draw_node(self, node, size, *args, **kwargs):
        """
        Draw a node as a sphere.
        """
        return Sphere(radius=size / 2.0, point=self.network.node_coordinates(node))

    def add_node(self, sphere, color):
        """
        Add one node to the viewer scene.
        """
        return self.viewer.scene.add(sphere,
                                     facecolor=color,
                                     linecolor=color,
                                     opacity=self.default_opacity)

    def add_nodes(self):
        """
        Add the nodes of the network to the viewer scene.
        """
        nodes = {}

        for node, sphere in self.collection_nodes.items():
            color = self.node_color[node]
            nodes[node] = self.add_node(sphere, color)

        return nodes

    def update_node(self, node, size):
        """
        Update a node in place.
        """
        sphere = self.collection_nodes[node]
        sphere.point = self.network.node_coordinates(node)
        sphere.radius = size / 2.0

        self.viewer_nodes[node].update(update_data=True)

    def update_nodes(self):
        """
        Update the nodes of the network.
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

        # shift start to make the arrow head touch the node the load is applied to
        start = add_vectors(xyz, scale_vector(vector, -scale))

        # shift start to account for half the size of the edge thickness
        widths = []
        for edge in self.network.node_edges(node):
            width = self.edge_width.get(edge)
            if not width:
                width = 0.0
            widths.append(width)

        start = add_vectors(start, scale_vector(normalize_vector(vector), -max(widths)))

        return self.draw_vector(vector, start, scale)

    def add_load(self, arrow, color):
        """
        Add one load to the viewer scene.
        """
        return self.viewer.scene.add(self.arrow_to_mesh(arrow),
                                     facecolor=color,
                                     linecolor=color,
                                     opacity=self.default_opacity)

    def add_loads(self):
        """
        Add the loads of the network to the viewer scene.
        """
        loads = {}

        for node, arrow in self.collection_loads.items():
            color = self.load_color
            if isinstance(color, dict):
                color = color[node]
            loads[node] = self.add_load(arrow, color)

        return loads

    def update_load(self, node, scale):
        """
        Update a load.

        An arrow changes topology as its vector changes, so the scene object is
        removed and re-added (clear-and-redraw) rather than mutated in place.
        """
        arrow = self.draw_load(node, scale)
        self.collection_loads[node] = arrow

        color = self.load_color
        if isinstance(color, dict):
            color = color[node]

        self.viewer.scene.remove(self.viewer_loads[node])
        self.viewer_loads[node] = self.add_load(arrow, color)

    def update_loads(self):
        """
        Update the loads of the network.
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

        # shift the starting point if the max force of connected edges is compressive
        connected_edges = list(network.node_edges(node))
        if len(connected_edges) == 0:
            return

        forces = [network.edge_force(e) for e in connected_edges]
        max_force = max(forces, key=lambda f: fabs(f))
        if max_force < 0.0:
            start = add_vectors(start, scale_vector(vector, scale))

        # reverse the vector to display the direction of the reaction forces
        return self.draw_vector(scale_vector(vector, -1.0), start, scale)

    def add_reaction(self, arrow, color):
        """
        Add one reaction force to the viewer scene.
        """
        return self.viewer.scene.add(self.arrow_to_mesh(arrow),
                                     facecolor=color,
                                     linecolor=color,
                                     opacity=self.default_opacity)

    def add_reactions(self):
        """
        Add the reaction forces of the network to the viewer scene.
        """
        reactions = {}

        for node, arrow in self.collection_reactions.items():
            color = self.reaction_color
            if isinstance(self.reaction_color, dict):
                color = color[node]
            reactions[node] = self.add_reaction(arrow, color)

        return reactions

    def update_reaction(self, node, scale):
        """
        Update a reaction force.

        An arrow changes topology as its vector changes, so the scene object is
        removed and re-added (clear-and-redraw) rather than mutated in place.
        """
        arrow = self.draw_reaction(node, scale)
        self.collection_reactions[node] = arrow

        color = self.reaction_color
        if isinstance(color, dict):
            color = color[node]

        self.viewer.scene.remove(self.viewer_reactions[node])
        self.viewer_reactions[node] = self.add_reaction(arrow, color)

    def update_reactions(self):
        """
        Update the reaction forces of the network.
        """
        for node in self.collection_reactions.keys():
            self.update_reaction(node, self.reaction_scale)

    # ==========================================================================
    # Helpers
    # ==========================================================================

    def draw_vector(self, vector, start, scale):
        """
        Build an arrow shape from a vector.
        """
        vector_scaled = scale_vector(vector, scale)

        return Arrow(position=start,
                     direction=vector_scaled,
                     head_portion=self.arrow_headportion,
                     head_width=self.arrow_headwidth,
                     body_width=self.arrow_bodywidth)

    @staticmethod
    def arrow_to_mesh(arrow):
        """
        Convert an :class:`Arrow` shape into a mesh the viewer can render.

        compas_viewer has no registered scene object for our custom ``Arrow``
        shape, but it renders a :class:`compas.datastructures.Mesh` directly.
        """
        return Mesh.from_vertices_and_faces(*arrow.to_vertices_and_faces())
