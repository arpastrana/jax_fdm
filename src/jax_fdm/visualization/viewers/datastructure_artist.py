from math import fabs

from compas.datastructures import Mesh
from compas.geometry import Cylinder
from compas.geometry import Line
from compas.geometry import Sphere
from compas.geometry import add_vectors
from compas.geometry import length_vector
from compas.geometry import normalize_vector
from compas.geometry import scale_vector
from jax_fdm.visualization.shapes import Arrow

__all__ = ["FDDatastructureViewerArtist"]


class FDDatastructureViewerArtist:
    """
    An artist that draws a force density datastructure to a :class:`compas_viewer.Viewer`.

    The artist builds plain COMPAS geometry (spheres for points,
    cylinders for edges, arrow meshes for load and reaction vectors) and pushes it
    into the viewer scene. It keeps a handle on every scene object so that an
    animation loop can mutate the geometry in place and re-read it with a single
    ``scene_object.update(update_data=True)``.

    This is the shared backend base for the network and mesh viewer artists; the
    node-vs-vertex vocabulary is resolved through the ``_point_*`` hooks provided
    by :class:`FDDatastructureArtist`.
    """
    default_opacity = 0.75
    arrow_bodywidth = 0.012
    arrow_headportion = 0.12
    arrow_headwidth = 0.04

    # Label of the per-category subgroup for the points in the viewer
    # tree. Subclasses override with "Nodes" / "Vertices".
    points_group_name = "Points"

    def __init__(self, datastructure, viewer, *args, **kwargs):
        super().__init__(datastructure, *args, **kwargs)
        self.viewer = viewer

        self.viewer_edges = {}
        self.viewer_points = {}
        self.viewer_loads = {}
        self.viewer_reactions = {}

        # Parent scene group and per-category subgroups, so the datastructure
        # shows up as a single foldable entity in the viewer tree instead of
        # hundreds of loose cylinders, spheres and arrow meshes.
        self.viewer_group = None
        self.viewer_groups = {}

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
        objects.update(self.viewer_points)
        objects.update(self.viewer_loads)
        objects.update(self.viewer_reactions)

        for obj in objects.values():
            yield obj

    # ==========================================================================
    # Add
    # ==========================================================================

    def add(self):
        """
        Add the points of the datastructure to the viewer scene.

        Every point is parented to a per-category subgroup (points,
        "Edges", "Reactions", "Loads") under a single group named after the
        datastructure, so the whole datastructure reads as one foldable entity in
        the viewer tree while each category (and each point) stays individually
        toggleable.
        """
        self.viewer_group = self.viewer.scene.add_group(name=self.datastructure.name or self.default_name)

        if self.collection_points:
            self.viewer_groups["points"] = self.viewer.scene.add_group(name=self.points_group_name, parent=self.viewer_group)
            self.viewer_points = self.add_points()
        if self.collection_edges:
            self.viewer_groups["edges"] = self.viewer.scene.add_group(name="Edges", parent=self.viewer_group)
            self.viewer_edges = self.add_edges()
        if self.collection_reactions:
            self.viewer_groups["reactions"] = self.viewer.scene.add_group(name="Reactions", parent=self.viewer_group)
            self.viewer_reactions = self.add_reactions()
        if self.collection_loads:
            self.viewer_groups["loads"] = self.viewer.scene.add_group(name="Loads", parent=self.viewer_group)
            self.viewer_loads = self.add_loads()

    @property
    def default_name(self):
        """
        The default name of the parent scene group.
        """
        return type(self.datastructure).__name__

    # ==========================================================================
    # Update
    # ==========================================================================

    def update(self):
        """
        Update the points of the datastructure drawn by this artist in place.
        """
        if self.viewer_points:
            self.update_points()
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
        start, end = self.datastructure.edge_coordinates(edge)
        line = Line(start, end)

        return Cylinder.from_line_and_radius(line, width / 2.0)

    def add_edge(self, cylinder, color, parent=None, name=None):
        """
        Add one edge to the viewer scene.
        """
        return self.viewer.scene.add(cylinder,
                                     facecolor=color,
                                     linecolor=color,
                                     opacity=self.default_opacity,
                                     parent=parent,
                                     name=name)

    def add_edges(self):
        """
        Add the edges of the datastructure to the viewer scene.
        """
        edges = {}
        parent = self.viewer_groups.get("edges")

        for edge, cylinder in self.collection_edges.items():
            color = self.edge_color[edge]
            u, v = edge
            edges[edge] = self.add_edge(cylinder, color, parent, f"Edge ({u}, {v})")

        return edges

    def update_edge(self, edge, width, color):
        """
        Update an edge in place.
        """
        cylinder = self.collection_edges[edge]

        start, end = self.datastructure.edge_coordinates(edge)
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
        Update the edges of the datastructure.
        """
        self.edge_color = self._init_edgecolor
        self.edge_width = self._init_edgewidth

        for edge in self.edges:
            width = self.edge_width[edge]
            color = self.edge_color[edge]
            self.update_edge(edge, width, color)

    # ==========================================================================
    # Points (nodes / vertices)
    # ==========================================================================

    def draw_point(self, point, size, *args, **kwargs):
        """
        Draw a point as a sphere.
        """
        return Sphere(radius=size / 2.0, point=self._point_coordinates(point))

    def add_point(self, sphere, color, parent=None, name=None):
        """
        Add one point to the viewer scene.
        """
        return self.viewer.scene.add(sphere,
                                     facecolor=color,
                                     linecolor=color,
                                     opacity=self.default_opacity,
                                     parent=parent,
                                     name=name)

    def add_points(self):
        """
        Add the points of the datastructure to the viewer scene.
        """
        points = {}
        parent = self.viewer_groups.get("points")

        for point, sphere in self.collection_points.items():
            color = self.point_color[point]
            points[point] = self.add_point(sphere, color, parent, self._point_label(point))

        return points

    def update_point(self, point, size):
        """
        Update a point in place.
        """
        sphere = self.collection_points[point]
        sphere.point = self._point_coordinates(point)
        sphere.radius = size / 2.0

        self.viewer_points[point].update(update_data=True)

    def update_points(self):
        """
        Update the points of the datastructure.
        """
        for point in self.collection_points.keys():
            size = self.point_size[point]
            self.update_point(point, size)

    # ==========================================================================
    # Loads
    # ==========================================================================

    def draw_load(self, point, scale, *args, **kwargs):
        """
        Draw a load vector at a point.
        """
        vector = self._point_load(point)

        if length_vector(vector) < self.load_tol:
            return

        xyz = self._point_coordinates(point)

        # shift start to make the arrow head touch the point the load is applied to
        start = add_vectors(xyz, scale_vector(vector, -scale))

        # shift start to account for half the size of the edge thickness
        widths = []
        for edge in self._point_edges(point):
            width = self.edge_width.get(edge)
            if not width:
                width = 0.0
            widths.append(width)

        start = add_vectors(start, scale_vector(normalize_vector(vector), -max(widths)))

        return self.draw_vector(vector, start, scale)

    def add_load(self, arrow, color, parent=None, name=None):
        """
        Add one load to the viewer scene.
        """
        return self.viewer.scene.add(self.arrow_to_mesh(arrow),
                                     facecolor=color,
                                     linecolor=color,
                                     opacity=self.default_opacity,
                                     parent=parent,
                                     name=name)

    def add_loads(self):
        """
        Add the loads of the datastructure to the viewer scene.
        """
        loads = {}
        parent = self.viewer_groups.get("loads")

        for point, arrow in self.collection_loads.items():
            color = self.load_color
            if isinstance(color, dict):
                color = color[point]
            loads[point] = self.add_load(arrow, color, parent, f"Load ({point})")

        return loads

    def update_load(self, point, scale):
        """
        Update a load.

        An arrow changes topology as its vector changes, so the scene object is
        removed and re-added (clear-and-redraw) rather than mutated in place.
        """
        arrow = self.draw_load(point, scale)
        self.collection_loads[point] = arrow

        color = self.load_color
        if isinstance(color, dict):
            color = color[point]

        self.viewer.scene.remove(self.viewer_loads[point])
        self.viewer_loads[point] = self.add_load(arrow, color, self.viewer_groups.get("loads"), f"Load ({point})")

    def update_loads(self):
        """
        Update the loads of the datastructure.
        """
        for point in self.collection_loads.keys():
            self.update_load(point, self.load_scale)

    # ==========================================================================
    # Reaction forces
    # ==========================================================================

    def draw_reaction(self, point, scale, *args, **kwargs):
        """
        Draw a reaction vector at a point.
        """
        vector = self._point_reaction(point)
        start = self._point_coordinates(point)

        if length_vector(vector) < self.reaction_tol:
            return

        # shift the starting point if the max force of connected edges is compressive
        connected_edges = list(self._point_edges(point))
        if len(connected_edges) == 0:
            return

        forces = [self.datastructure.edge_force(e) for e in connected_edges]
        max_force = max(forces, key=lambda f: fabs(f))
        if max_force < 0.0:
            start = add_vectors(start, scale_vector(vector, scale))

        # reverse the vector to display the direction of the reaction forces
        return self.draw_vector(scale_vector(vector, -1.0), start, scale)

    def add_reaction(self, arrow, color, parent=None, name=None):
        """
        Add one reaction force to the viewer scene.
        """
        return self.viewer.scene.add(self.arrow_to_mesh(arrow),
                                     facecolor=color,
                                     linecolor=color,
                                     opacity=self.default_opacity,
                                     parent=parent,
                                     name=name)

    def add_reactions(self):
        """
        Add the reaction forces of the datastructure to the viewer scene.
        """
        reactions = {}
        parent = self.viewer_groups.get("reactions")

        for point, arrow in self.collection_reactions.items():
            color = self.reaction_color
            if isinstance(self.reaction_color, dict):
                color = color[point]
            reactions[point] = self.add_reaction(arrow, color, parent, f"Reaction ({point})")

        return reactions

    def update_reaction(self, point, scale):
        """
        Update a reaction force.

        An arrow changes topology as its vector changes, so the scene object is
        removed and re-added (clear-and-redraw) rather than mutated in place.
        """
        arrow = self.draw_reaction(point, scale)
        self.collection_reactions[point] = arrow

        color = self.reaction_color
        if isinstance(color, dict):
            color = color[point]

        self.viewer.scene.remove(self.viewer_reactions[point])
        self.viewer_reactions[point] = self.add_reaction(arrow, color, self.viewer_groups.get("reactions"), f"Reaction ({point})")

    def update_reactions(self):
        """
        Update the reaction forces of the datastructure.
        """
        for point in self.collection_reactions.keys():
            self.update_reaction(point, self.reaction_scale)

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
