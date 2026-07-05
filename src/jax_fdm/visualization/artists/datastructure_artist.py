from collections.abc import Iterable
from math import fabs

from compas.colors import Color
from compas.colors import ColorMap
from compas.itertools import remap_values

__all__ = ["FDDatastructureArtist"]


class FDDatastructureArtist:
    """
    The base artist to display a force density datastructure across contexts.

    This is a plain, COMPAS-free base class shared by the network and the mesh
    artists: it owns the force-density-specific computation (point and edge
    colors, edge widths, load and reaction vectors) and exposes it as
    backend-neutral dictionaries. Concrete backends subclass it and implement the
    abstract ``draw_*`` methods to turn that data into their own scene objects.

    The points of the datastructure (nodes for a network, vertices for a
    mesh) are accessed through the ``_point_*`` hooks so that the shared logic
    stays free of the vertex-vs-node terminology. The edge accessors
    (``edge_coordinates``, ``edge_force``, ``edge_forcedensity``) already live on
    the shared :class:`jax_fdm.datastructures.FDDatastructure`, so they are used
    directly.
    """
    default_edgecolor = Color.teal()
    default_nodecolor = Color.grey().lightened(factor=100)
    default_nodesupportcolor = Color.from_rgb255(0, 150, 10)

    default_loadcolor = Color.from_rgb255(0, 150, 10)
    default_reactioncolor = Color.pink()

    default_tensioncolor = Color.from_rgb255(227, 6, 75)
    default_compressioncolor = Color.from_rgb255(12, 119, 184)

    default_fdcolormap = ColorMap.from_mpl("viridis")
    default_forcecolormap = ColorMap.from_three_colors(Color.from_rgb255(12, 119, 184),
                                                       Color.grey().lightened(50),
                                                       Color.from_rgb255(227, 6, 75))

    default_nodesize = 0.1
    default_edgewidth = (0.01, 0.1)
    default_loadscale = 1.0
    default_reactionscale = 1.0
    default_loadtol = 1e-3
    default_reactiontol = 1e-3

    def __init__(self,
                 datastructure,
                 points=None,
                 edges=None,
                 nodecolor=None,
                 edgecolor=None,
                 nodesize=None,
                 edgewidth=None,
                 loadcolor=None,
                 loadscale=None,
                 loadtol=None,
                 reactioncolor=None,
                 reactionscale=None,
                 reactiontol=None,
                 show_nodes=False,
                 show_edges=True,
                 show_loads=True,
                 show_reactions=True,
                 show_supports=True,
                 *args,
                 **kwargs):
        self.datastructure = datastructure

        # Point (node/vertex) and edge iterables, optionally filtered
        # (defaults to all).
        self.points = list(points) if points is not None else list(self._points())
        self.edges = list(edges) if edges is not None else list(datastructure.edges())

        self._point_xyz = None
        self._point_color = None
        self._edge_color = None
        self._edge_width = None

        self._default_loadcolor = None
        self._default_reactioncolor = None
        self._default_nodesupportcolor = None
        self._default_tensioncolor = None
        self._default_compressioncolor = None

        self._point_size = None

        self.point_size = nodesize
        self.edge_width = edgewidth

        self.point_color = nodecolor
        self.edge_color = edgecolor

        self.load_color = loadcolor or self.default_loadcolor

        if edgecolor == "force":
            self.default_reactioncolor = Color.from_rgb255(0, 150, 10)
        self.reaction_color = reactioncolor or self.default_reactioncolor

        self.load_scale = loadscale or self.default_loadscale
        self.load_tol = loadtol or self.default_loadtol

        self.reaction_scale = reactionscale or self.default_reactionscale
        self.reaction_tol = reactiontol or self.default_reactiontol

        self.show_points = show_nodes
        self.show_edges = show_edges
        self.show_loads = show_loads
        self.show_reactions = show_reactions
        self.show_supports = show_supports

        self.collection_edges = None
        self.collection_points = None
        self.collection_loads = None
        self.collection_reactions = None

        self._init_edgecolor = edgecolor
        self._init_edgewidth = edgewidth

    # ==========================================================================
    # Point hooks (implemented by subclasses to bridge node/vertex vocabulary)
    # ==========================================================================

    def _points(self):
        """
        Iterate over the keys of the points (nodes or vertices).
        """
        raise NotImplementedError

    def _point_coordinates(self, key):
        """
        The xyz coordinates of a point.
        """
        raise NotImplementedError

    def _point_load(self, key):
        """
        The load vector applied at a point.
        """
        raise NotImplementedError

    def _point_reaction(self, key):
        """
        The reaction force vector at a point.
        """
        raise NotImplementedError

    def _point_edges(self, key):
        """
        The edges connected to a point.
        """
        raise NotImplementedError

    def _point_is_support(self, key):
        """
        Test whether a point is a support.
        """
        raise NotImplementedError

    def _point_label(self, key):
        """
        A human-readable label for a point (used by scene backends).
        """
        raise NotImplementedError

    # ==========================================================================
    # Draw
    # ==========================================================================

    def draw(self):
        """
        Draw everything.
        """
        if self.show_edges:
            self.collection_edges = self.draw_edges()
        if self.show_points:
            self.collection_points = self.draw_points()
        if self.show_loads:
            self.collection_loads = self.draw_loads()
        if self.show_reactions:
            self.collection_reactions = self.draw_reactions()

    # ==========================================================================
    # Draw collections
    # ==========================================================================

    def draw_points(self):
        """
        Draw the points of the datastructure.
        """
        points = {}
        for point in self.points:
            size = self.point_size[point]
            color = self.point_color[point]
            points[point] = self.draw_point(point, size, color)

        return points

    def draw_edges(self):
        """
        Draw the edges of the datastructure.
        """
        edges = {}

        for edge in self.edges:
            width = self.edge_width[edge]
            color = self.edge_color[edge]
            edges[edge] = self.draw_edge(edge, width, color)

        return edges

    def draw_loads(self):
        """
        Draw the loads at the points of the datastructure.
        """
        loads = {}

        for point in self.points:
            load = self.draw_load(point, self.load_scale, self.load_color)
            if load:
                loads[point] = load

        return loads

    def draw_reactions(self):
        """
        Draw the reactions at the points of the datastructure.
        """
        reactions = {}

        for point in self.points:
            reaction = self.draw_reaction(point, self.reaction_scale, self.reaction_color)
            if reaction:
                reactions[point] = reaction

        return reactions

    # ==========================================================================
    # Draw points
    # ==========================================================================

    def draw_point(self, point, size, color):
        """
        Draw a point.
        """
        raise NotImplementedError

    def draw_edge(self, edge, width, color):
        """
        Draw an edge.
        """
        raise NotImplementedError

    def draw_load(self, point, scale, color):
        """
        Draw a load.
        """
        raise NotImplementedError

    def draw_reaction(self, point, scale, color):
        """
        Draw a load.
        """
        raise NotImplementedError

    def clear_points(self):
        """
        Clear the points.
        """
        pass

    def clear_edges(self):
        """
        Clear the edges.
        """
        pass

    # ==========================================================================
    # Update points
    # ==========================================================================

    def update(self, eqstate):
        """
        Update the attributes of the datastructure based on an equilibrium state.
        """
        raise NotImplementedError

    # ==========================================================================
    # Properties
    # ==========================================================================

    @property
    def point_xyz(self):
        if not self._point_xyz:
            self._point_xyz = {
                point: self._point_coordinates(point)
                for point in self._points()
            }
        return self._point_xyz

    @point_xyz.setter
    def point_xyz(self, point_xyz):
        self._point_xyz = point_xyz

    @property
    def edge_color(self):
        """
        The edge colors.
        """
        if not self._edge_color:
            self._edge_color = {edge: self.default_edgecolor for edge in self.edges}
        return self._edge_color

    @edge_color.setter
    def edge_color(self, color):
        if isinstance(color, dict):
            self._edge_color = color

        elif isinstance(color, Color):
            self._edge_color = {edge: color for edge in self.edges}

        elif isinstance(color, str):
            datastructure = self.datastructure

            if color == "fd":
                cmap = self.default_fdcolormap
                values = [fabs(datastructure.edge_forcedensity(edge)) for edge in self.edges]
                try:
                    ratios = remap_values(values)
                except ZeroDivisionError:
                    ratios = [0.0] * len(self.edges)

                self._edge_color = {edge: cmap(ratio) for edge, ratio in zip(self.edges, ratios)}

            elif color == "force":
                cmap = self.default_forcecolormap

                edge_color = {}
                for edge in self.edges:
                    force = datastructure.edge_force(edge)
                    _color = self.default_tensioncolor
                    if force <= 0.0:
                        _color = self.default_compressioncolor
                    edge_color[edge] = _color

                self._edge_color = edge_color

    @property
    def point_color(self):
        """
        The point colors.
        """
        if self._point_color:
            return self._point_color

        point_color = {}

        for point in self.points:
            color = self.default_nodecolor
            if self._point_is_support(point):
                color = self.default_nodesupportcolor
            point_color[point] = color

        self._point_color = point_color

        return self._point_color

    @point_color.setter
    def point_color(self, color):
        if isinstance(color, dict):
            self._point_color = color
        elif isinstance(color, Color):
            self._point_color = {point: color for point in self.points}

    @property
    def point_size(self):
        """
        The size of the points.
        """
        if not self._point_size:
            self.point_size = self.default_nodesize
        return self._point_size

    @point_size.setter
    def point_size(self, size):
        if isinstance(size, dict):
            self._point_size = size
        elif isinstance(size, (int, float)):
            self._point_size = {point: size for point in self.points}

    @property
    def edge_width(self):
        """
        The width of the edges.
        """
        if not self._edge_width:
            self.edge_width = self.default_edgewidth
        return self._edge_width

    @edge_width.setter
    def edge_width(self, width):
        if isinstance(width, dict):
            self._edge_width = width

        elif isinstance(width, (int, float)):
            self._edge_width = {edge: width for edge in self.edges}

        elif isinstance(width, Iterable) and len(width) == 2:
            width_min, width_max = width

            if not self.edges:
                return

            forces = [fabs(self.datastructure.edge_force(edge)) for edge in self.edges]

            if min(forces) == max(forces):
                widths = [width_max] * len(self.edges)
            else:
                try:
                    widths = remap_values(forces, width_min, width_max)
                except ZeroDivisionError:
                    widths = [self.default_edgewidth][0] * len(self.edges)

            self._edge_width = {edge: width for edge, width in zip(self.edges, widths)}
