import numpy as np
from compas_viewer.scene import BufferGeometry

from jax_fdm.visualization.artists import FDShapeArtist
from jax_fdm.visualization.viewers.buffers import arrows_buffer
from jax_fdm.visualization.viewers.buffers import cylinders_buffer
from jax_fdm.visualization.viewers.buffers import spheres_buffer

__all__ = ["FDDatastructureViewerArtist"]


class FDDatastructureViewerArtist(FDShapeArtist):
    """
    An artist that draws a force density datastructure to a :class:`compas_viewer.Viewer`.

    Every element category (edges as cylinders, points as spheres, loads and
    reactions as arrows) is batched into a single triangle-soup
    :class:`compas_viewer.scene.BufferGeometry`, so a whole datastructure
    costs a handful of scene objects instead of two per element, and an
    animation loop updates one render buffer per category in place.

    The soup topology is frozen at add time (the render buffers never change
    size): arrows are allocated for every candidate point and collapse to a
    degenerate soup at their anchor while below tolerance, and edge cylinders
    keep their vertex count as widths and colors change.

    This is the shared backend base for the network and mesh viewer artists;
    the node-vs-vertex vocabulary is resolved through the ``_point_*`` hooks
    provided by :class:`FDDatastructureArtist`.
    """
    default_opacity = 0.75

    shape_u = 16
    arrow_u = 8

    # Label of the points scene object in the viewer tree.
    # Subclasses override with "Nodes" / "Vertices".
    points_group_name = "Points"

    def __init__(self, datastructure, viewer, *args, name=None, **kwargs):
        super().__init__(datastructure, *args, **kwargs)
        self.viewer = viewer
        self.name = name

        self.viewer_edges = None
        self.viewer_points = None
        self.viewer_loads = None
        self.viewer_reactions = None

        self.viewer_group = None

        # Candidate point lists for the arrow categories, frozen at draw time
        # so the soup membership never changes across updates.
        self._load_points = None
        self._reaction_points = None

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def objects(self):
        """
        Yield the scene objects drawn by this artist.
        """
        for obj in (self.viewer_edges,
                    self.viewer_points,
                    self.viewer_loads,
                    self.viewer_reactions):
            if obj is not None:
                yield obj

    @property
    def default_name(self):
        """
        The default name of the parent scene group.
        """
        return type(self.datastructure).__name__

    # ==========================================================================
    # Draw one element (lightweight; tessellation happens in the batch builders)
    # ==========================================================================

    def draw_edge(self, edge, width, *args, **kwargs):
        """
        Draw an edge as its axis endpoints and radius.
        """
        start, end = self.datastructure.edge_coordinates(edge)
        return (start, end, width / 2.0)

    def draw_point(self, point, size, *args, **kwargs):
        """
        Draw a point as its center and radius.
        """
        return (self._point_coordinates(point), size / 2.0)

    # ==========================================================================
    # Add
    # ==========================================================================

    def add(self, group=None):
        """
        Add the elements of the datastructure to the viewer scene.

        Every category becomes one buffer scene object parented to a single
        group, so the datastructure reads as one foldable entity in the viewer
        tree while each category stays individually toggleable. The group takes
        the ``name`` passed to the artist (e.g. via ``viewer.add(data, name=...)``),
        falling back to the datastructure's own name and finally to its type name.

        An existing scene group can be supplied via ``group`` to host the
        category objects directly (used by the registered scene-object adapters).
        """
        if group is not None:
            self.viewer_group = group
        else:
            self.viewer_group = self.viewer.scene.add_group(name=self.name or self.datastructure.name or self.default_name)

        if self.collection_points:
            self.viewer_points = self.add_buffer(*self.points_arrays(), self.points_group_name)
        if self.collection_edges:
            self.viewer_edges = self.add_buffer(*self.edges_arrays(), "Edges")
        if self.collection_reactions is not None:
            self._reaction_points = [point for point in self.points if list(self._point_edges(point))]
            self.viewer_reactions = self.add_buffer(*self.reactions_arrays(), "Reactions")
        if self.collection_loads is not None:
            self._load_points = list(self.points)
            self.viewer_loads = self.add_buffer(*self.loads_arrays(), "Loads")

    def add_buffer(self, positions, colors, name):
        """
        Add one category batch to the viewer scene as a buffer object.
        """
        geometry = BufferGeometry(faces=positions, facecolor=colors)

        return self.viewer.scene.add(geometry,
                                     name=name,
                                     parent=self.viewer_group,
                                     opacity=self.default_opacity,
                                     show_points=False,
                                     show_lines=False)

    # ==========================================================================
    # Update
    # ==========================================================================

    def update(self):
        """
        Update the render buffers of the datastructure drawn by this artist in place.
        """
        # Re-remap edge colors and widths against the updated force densities
        # and forces. Widths must precede loads: the load start shift depends
        # on the width of the connected edges.
        self.edge_color = self._init_edgecolor
        self.edge_width = self._init_edgewidth

        if self.viewer_points is not None:
            self.collection_points = self.draw_points()
            self.update_buffer(self.viewer_points, *self.points_arrays())
        if self.viewer_edges is not None:
            self.collection_edges = self.draw_edges()
            self.update_buffer(self.viewer_edges, *self.edges_arrays())
        if self.viewer_reactions is not None:
            self.update_buffer(self.viewer_reactions, *self.reactions_arrays())
        if self.viewer_loads is not None:
            self.update_buffer(self.viewer_loads, *self.loads_arrays())

    @staticmethod
    def update_buffer(obj, positions, colors):
        """
        Write one category batch into its render buffer in place.
        """
        obj.buffergeometry.faces = positions
        obj.buffergeometry.facecolor = colors
        obj.update(update_data=True)

    # ==========================================================================
    # Category batches
    # ==========================================================================

    def edges_arrays(self):
        """
        The triangle soup of the edges of the datastructure.
        """
        edges = list(self.collection_edges.keys())

        starts, ends, radii = [], [], []
        for edge in edges:
            start, end, radius = self.collection_edges[edge]
            starts.append(start)
            ends.append(end)
            radii.append(radius)

        colors = np.array([self.edge_color[edge].rgba for edge in edges])

        return cylinders_buffer(starts, ends, radii, colors, u=self.shape_u)

    def points_arrays(self):
        """
        The triangle soup of the points of the datastructure.
        """
        points = list(self.collection_points.keys())

        centers, radii = [], []
        for point in points:
            center, radius = self.collection_points[point]
            centers.append(center)
            radii.append(radius)

        colors = np.array([self.point_color[point].rgba for point in points])

        return spheres_buffer(centers, radii, colors, u=self.shape_u, v=self.shape_u)

    def loads_arrays(self):
        """
        The triangle soup of the load vectors of the datastructure.
        """
        return self.vectors_arrays(self._load_points, self.draw_load, self.load_scale, self.load_color)

    def reactions_arrays(self):
        """
        The triangle soup of the reaction vectors of the datastructure.
        """
        return self.vectors_arrays(self._reaction_points, self.draw_reaction, self.reaction_scale, self.reaction_color)

    def vectors_arrays(self, points, draw_vector, scale, color):
        """
        The triangle soup of one arrow category over its candidate points.

        Every candidate point gets a soup slot: points whose vector is below
        tolerance yield a degenerate arrow at their coordinates, so the soup
        size stays constant while vectors grow, shrink or vanish across updates.
        """
        anchors, vectors, colors = [], [], []

        for point in points:
            arrow = draw_vector(point, scale)
            if arrow is None:
                anchors.append(self._point_coordinates(point))
                vectors.append((0.0, 0.0, 0.0))
            else:
                anchors.append(arrow.position)
                vectors.append(arrow.direction)

            _color = color[point] if isinstance(color, dict) else color
            colors.append(_color.rgba)

        return arrows_buffer(anchors,
                             vectors,
                             np.array(colors) if colors else [],
                             head_portion=self.arrow_headportion,
                             head_width=self.arrow_headwidth,
                             body_width=self.arrow_bodywidth,
                             u=self.arrow_u)
