from math import hypot
from typing import Any

from compas_plotter.scene import GraphObject
from compas_plotter.scene import MeshObject
from compas_plotter.scene import PlotterSceneObject
from compas_plotter.scene.plotterobject import to_rgb
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
from matplotlib.patches import FancyArrow

from compas.colors import Color
from compas.scene import register
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization.style import COLOR_LOAD
from jax_fdm.visualization.style import LOAD_SCALE
from jax_fdm.visualization.style import LOAD_TOL
from jax_fdm.visualization.style import REACTION_SCALE
from jax_fdm.visualization.style import REACTION_TOL
from jax_fdm.visualization.style import EdgeColorSpec
from jax_fdm.visualization.style import EdgeWidthSpec
from jax_fdm.visualization.style import PointColorSpec
from jax_fdm.visualization.style import PointSizeSpec
from jax_fdm.visualization.style import edge_colors
from jax_fdm.visualization.style import edge_widths
from jax_fdm.visualization.style import load_arrows
from jax_fdm.visualization.style import point_colors
from jax_fdm.visualization.style import reaction_arrows
from jax_fdm.visualization.style import reaction_color_default

__all__ = [
    "FDPlotterObject",
    "FDNetworkPlotterObject",
    "FDMeshPlotterObject",
    "register_plotter_scene_objects",
]


# 2D display defaults. Matplotlib linewidths live in points, not world units,
# so the plotter remaps edge forces into its own width bounds instead of the
# world-unit cylinder radii of the 3D backends. The arrow proportions are
# fractions of the arrow length, carried over from the 1.x plotter artist:
# chunkier than the 3D fractions in style.py, which read too thin as flat
# polygons.
EDGE_WIDTH_2D = (0.5, 5.0)
ARROW_BODYWIDTH_2D = 0.024
ARROW_HEADPORTION_2D = 0.2
ARROW_HEADWIDTH_2D = 0.08


class FDPlotterObject(PlotterSceneObject):
    """
    A scene object that draws a force density datastructure in a plotter.

    The force density styling (edge colors and widths from forces, support
    colors, load and reaction arrows) is resolved every draw, so the redraw
    cycle of the plotter animation loop picks up datastructure changes.

    Loads and reactions are drawn as data-space arrow polygons whose head and
    body are proportional to the arrow length, batched into one matplotlib
    collection per category. Out-of-plane arrows project to zero length and
    are skipped.

    Subclasses pair this class with the matching compas_plotter drawable and
    resolve the point vocabulary (a network addresses its points as nodes, a
    mesh as vertices) by implementing the ``point_*`` methods and by exposing
    it as constructor keyword arguments (``nodes``/``nodecolor``/``nodesize``/
    ``show_nodes`` on a network, ``vertices``/``vertexcolor``/``vertexsize``/
    ``show_vertices`` on a mesh) that map onto the neutral point parameters
    here.

    The ``points`` and ``edges`` filters drive the styling and the arrows;
    the point markers themselves are drawn by the compas_plotter base for
    all points of the datastructure.
    """

    # The upstream attribute names of the point color setting and the point
    # display flag, resolved by the subclasses to their vocabulary.
    point_color_attr: str | None = None
    point_show_attr: str | None = None

    def __init__(
        self,
        item: FDNetwork | FDMesh | None = None,
        points: list[int] | None = None,
        edges: list[tuple[int, int]] | None = None,
        pointcolor: PointColorSpec = None,
        edgecolor: EdgeColorSpec = None,
        edgewidth: EdgeWidthSpec = None,
        loadcolor: Color | None = None,
        loadscale: float | None = None,
        loadtol: float | None = None,
        reactioncolor: Color | None = None,
        reactionscale: float | None = None,
        reactiontol: float | None = None,
        show_points: bool = False,
        show_edges: bool = True,
        show_loads: bool = True,
        show_reactions: bool = True,
        show_supports: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.pop("sceneobject_type", None)
        super().__init__(item=item, **kwargs)

        # The scene registry always dispatches a real datastructure to this
        # constructor; the Optional in the signature only matches the base
        # class default.
        assert item is not None
        self.datastructure: FDNetwork | FDMesh = item

        self.points: list[int] = (
            list(points) if points is not None else list(self.point_keys())
        )
        self.edges: list[tuple[int, int]] = (
            list(edges) if edges is not None else list(self.datastructure.edges())
        )

        # The point-edge adjacency is cached once at construction, so drawing
        # never re-derives it from the datastructure (Mesh.vertex_edges scans
        # all mesh edges per call).
        self.adjacency = {point: list(self.point_edges(point)) for point in self.points}

        self.pointcolor_spec = pointcolor
        self.edgecolor_spec = edgecolor
        self.edgewidth_spec = edgewidth
        self.show_supports = show_supports if show_supports is not None else True

        self.load_color = loadcolor or COLOR_LOAD
        self.load_scale = loadscale or LOAD_SCALE
        self.load_tol = loadtol or LOAD_TOL

        self.reaction_color = reactioncolor or reaction_color_default(edgecolor)
        self.reaction_scale = reactionscale or REACTION_SCALE
        self.reaction_tol = reactiontol or REACTION_TOL

        self.show_loads = show_loads if show_loads is not None else True
        self.show_reactions = show_reactions if show_reactions is not None else True

        # subclasses always set point_show_attr to a str before instantiation
        setattr(self, self.point_show_attr, bool(show_points))  # pyright: ignore[reportArgumentType]
        self.show_edges = show_edges if show_edges is not None else True

    # ==========================================================================
    # Point vocabulary
    # ==========================================================================

    def point_keys(self) -> list[int]:
        raise NotImplementedError

    def point_coordinates(self, key: int) -> list[float]:
        raise NotImplementedError

    def point_load(self, key: int) -> list[float]:
        raise NotImplementedError

    def point_reaction(self, key: int) -> list[float]:
        raise NotImplementedError

    def point_edges(self, key: int) -> list[tuple[int, int]]:
        raise NotImplementedError

    def point_is_support(self, key: int) -> bool:
        raise NotImplementedError

    def point_marker_radius(self) -> float:
        """
        The radius of the point markers in data units, for the arrow clearance.

        Zero when the markers are hidden, so the arrows touch the bare points.
        """
        raise NotImplementedError

    # ==========================================================================
    # Style
    # ==========================================================================

    def resolve_style(self) -> None:
        """
        Derive the per-element styling from the current datastructure state.

        The colors are assigned onto the upstream per-element color settings,
        so the inherited drawing methods pick them up; the per-edge widths are
        kept apart because the upstream edge pass only broadcasts one width.
        """
        datastructure = self.datastructure

        width = (
            self.edgewidth_spec if self.edgewidth_spec is not None else EDGE_WIDTH_2D
        )
        self.edge_width = edge_widths(datastructure, self.edges, width)
        self.edgecolor = edge_colors(datastructure, self.edges, self.edgecolor_spec)

        # Free points default to white on the plotter, matching the white
        # markers of the compas_plotter canvas rather than the grey of the
        # shaded 3D backends.
        is_support = (
            self.point_is_support if self.show_supports else (lambda key: False)
        )
        # point_colors treats a str spec as an unrecognized dict/Color and falls
        # back to the default.
        colors = point_colors(
            self.points,
            is_support,
            self.pointcolor_spec,  # pyright: ignore[reportArgumentType]
            default=Color.white(),
        )
        # subclasses always set point_color_attr to a str before instantiation
        setattr(self, self.point_color_attr, colors)  # pyright: ignore[reportArgumentType]

    # ==========================================================================
    # Draw
    # ==========================================================================

    def draw(self) -> list[Any]:
        """
        Draw the datastructure and its load and reaction arrows.
        """
        self.resolve_style()

        super().draw()

        if self.show_loads:
            anchors, vectors = self._load_arrow_data()
            self._draw_arrows(anchors, vectors, self.load_color)

        if self.show_reactions:
            anchors, vectors = self._reaction_arrow_data()
            self._draw_arrows(anchors, vectors, self.reaction_color)

        return self._mpl_objects

    def viewdata(self) -> list[list[float]]:
        """
        The 2D extents of the points and the arrows, for the plotter zoom.
        """
        data = [self.point_coordinates(point)[:2] for point in self.points]

        categories = []
        if self.show_loads:
            categories.append(self._load_arrow_data())
        if self.show_reactions:
            categories.append(self._reaction_arrow_data())

        for anchors, vectors in categories:
            for anchor, vector in zip(anchors, vectors):
                data.append([anchor[0], anchor[1]])
                data.append([anchor[0] + vector[0], anchor[1] + vector[1]])

        return data

    def _draw_edges(self, point_xyz: dict[int, list[float]]) -> None:
        """
        Draw the edges with their force density styling.

        Replaces the upstream edge pass to draw only the edge selection and
        to pass the per-edge widths, which the upstream pass broadcasts from
        a single scalar.
        """
        lines, colors, widths = [], [], []
        for edge in self.edges:
            u, v = edge
            lines.append([point_xyz[u][:2], point_xyz[v][:2]])
            colors.append(to_rgb(self.edgecolor[edge]))
            widths.append(self.edge_width[edge])

        collection = LineCollection(
            lines,
            linewidths=widths,
            colors=colors,
            zorder=self.zorder + 10,
        )
        self.axes.add_collection(collection)
        self._mpl_objects.append(collection)

    def _draw_arrows(
        self,
        anchors: list[list[float]],
        vectors: list[list[float]],
        color: Color,
    ) -> None:
        """
        Batch one arrow category into a single matplotlib collection.
        """
        patches = []
        for anchor, vector in zip(anchors, vectors):
            dx, dy = vector[0], vector[1]
            length = hypot(dx, dy)
            if not length:
                continue

            patches.append(
                FancyArrow(
                    anchor[0],
                    anchor[1],
                    dx,
                    dy,
                    width=ARROW_BODYWIDTH_2D * length,
                    head_length=ARROW_HEADPORTION_2D * length,
                    head_width=ARROW_HEADWIDTH_2D * length,
                    length_includes_head=True,
                    lw=0.0,
                ),
            )
        if not patches:
            return

        collection = PatchCollection(
            patches,
            facecolor=to_rgb(color),
            edgecolor="none",
            zorder=5000,
        )
        self.axes.add_collection(collection)
        self._mpl_objects.append(collection)

    def _load_arrow_data(self) -> tuple[list[list[float]], list[list[float]]]:
        """
        The anchor and vector of the load arrow at every point.

        The arrows clear the point markers by their radius.
        """
        origins = [self.point_coordinates(point) for point in self.points]
        loads = [self.point_load(point) for point in self.points]
        clearances = [self.point_marker_radius()] * len(origins)

        return load_arrows(origins, loads, clearances, self.load_scale, self.load_tol)

    def _reaction_arrow_data(self) -> tuple[list[list[float]], list[list[float]]]:
        """
        The anchor and vector of the reaction arrow at every connected point.

        The arrows clear the point markers by their radius.
        """
        points = [point for point in self.points if self.adjacency[point]]
        origins = [self.point_coordinates(point) for point in points]
        reactions = [self.point_reaction(point) for point in points]
        forces = [
            [self.datastructure.edge_force(edge) for edge in self.adjacency[point]]
            for point in points
        ]
        clearances = [self.point_marker_radius()] * len(origins)

        return reaction_arrows(
            origins,
            reactions,
            forces,
            self.reaction_scale,
            self.reaction_tol,
            clearances=clearances,
        )


class FDNetworkPlotterObject(FDPlotterObject, GraphObject):
    """
    A scene object that draws a force density network in a plotter.

    The network points are filtered with the ``nodes`` keyword argument and
    styled with ``nodecolor``, ``nodesize`` and ``show_nodes``, matching the
    datastructure vocabulary.
    """

    point_color_attr = "nodecolor"
    point_show_attr = "show_nodes"

    # Narrows the base class's FDNetwork | FDMesh attribute to the type this
    # subclass actually holds, so the network-vocabulary accessors below
    # type-check against the right datastructure.
    datastructure: FDNetwork

    def __init__(
        self,
        item: FDNetwork | None = None,
        nodes: list[int] | None = None,
        nodecolor: PointColorSpec = None,
        nodesize: PointSizeSpec = None,
        show_nodes: bool | None = None,
        **kwargs: Any,
    ) -> None:
        # Map the node vocabulary onto the neutral point parameters of the
        # base, dropping injected None defaults so they never clobber the
        # node keywords. The marker size stays in the upstream vocabulary:
        # it is an on-screen size in points, not a world-space sphere radius.
        points = kwargs.pop("points", None)
        pointcolor = kwargs.pop("pointcolor", None)
        pointsize = kwargs.pop("pointsize", None)
        show_points = kwargs.pop("show_points", None)

        size = nodesize if nodesize is not None else pointsize
        if size is not None:
            kwargs["nodesize"] = size

        super().__init__(
            item=item,
            points=nodes if nodes is not None else points,
            pointcolor=nodecolor if nodecolor is not None else pointcolor,
            show_points=show_nodes if show_nodes is not None else show_points,
            **kwargs,
        )

    def point_keys(self) -> list[int]:
        # the data=False getter always yields plain node keys
        return self.datastructure.nodes()  # pyright: ignore[reportReturnType]

    def point_coordinates(self, key: int) -> list[float]:
        return self.datastructure.node_coordinates(key)

    def point_load(self, key: int) -> list[float]:
        # the getter-mode call always returns a list
        return self.datastructure.node_load(key)  # pyright: ignore[reportReturnType]

    def point_reaction(self, key: int) -> list[float]:
        return self.datastructure.node_reaction(key)

    def point_edges(self, key: int) -> list[tuple[int, int]]:
        return self.datastructure.node_edges(key)

    def point_is_support(self, key: int) -> bool:
        return self.datastructure.is_node_support(key)

    def point_marker_radius(self) -> float:
        if not self.show_nodes:
            return 0.0
        return self._node_radius()


class FDMeshPlotterObject(FDPlotterObject, MeshObject):
    """
    A scene object that draws a force density mesh in a plotter.

    On top of the shared edge/load/reaction categories, the mesh faces are
    drawn by the inherited compas_plotter face pass.

    The mesh points are filtered with the ``vertices`` keyword argument and
    styled with ``vertexcolor``, ``vertexsize`` and ``show_vertices``,
    matching the datastructure vocabulary.
    """

    point_color_attr = "vertexcolor"
    point_show_attr = "show_vertices"

    # Narrows the base class's FDNetwork | FDMesh attribute to the type this
    # subclass actually holds, so the vertex-vocabulary accessors below
    # type-check against the right datastructure.
    datastructure: FDMesh

    def __init__(
        self,
        item: FDMesh | None = None,
        vertices: list[int] | None = None,
        vertexcolor: PointColorSpec = None,
        vertexsize: PointSizeSpec = None,
        show_vertices: bool | None = None,
        **kwargs: Any,
    ) -> None:
        # Map the vertex vocabulary onto the neutral point parameters of the
        # base, dropping injected None defaults so they never clobber the
        # vertex keywords. The marker size stays in the upstream vocabulary:
        # it is an on-screen size in points, not a world-space sphere radius.
        points = kwargs.pop("points", None)
        pointcolor = kwargs.pop("pointcolor", None)
        pointsize = kwargs.pop("pointsize", None)
        show_points = kwargs.pop("show_points", None)

        size = vertexsize if vertexsize is not None else pointsize
        if size is not None:
            kwargs["vertexsize"] = size

        super().__init__(
            item=item,
            points=vertices if vertices is not None else points,
            pointcolor=vertexcolor if vertexcolor is not None else pointcolor,
            show_points=show_vertices if show_vertices is not None else show_points,
            **kwargs,
        )

    def point_keys(self) -> list[int]:
        # the data=False getter always yields plain vertex keys
        return self.datastructure.vertices()  # pyright: ignore[reportReturnType]

    def point_coordinates(self, key: int) -> list[float]:
        # the getter-mode call always returns a list
        return self.datastructure.vertex_coordinates(key)

    def point_load(self, key: int) -> list[float]:
        # the getter-mode call always returns a list
        return self.datastructure.vertex_load(key)  # pyright: ignore[reportReturnType]

    def point_reaction(self, key: int) -> list[float]:
        return self.datastructure.vertex_reaction(key)

    def point_edges(self, key: int) -> list[tuple[int, int]]:
        return self.datastructure.vertex_edges(key)

    def point_is_support(self, key: int) -> bool:
        return self.datastructure.is_vertex_support(key)

    def point_marker_radius(self) -> float:
        # Mirrors the radius of the vertex circles drawn by the
        # compas_plotter mesh scene object.
        if not self.show_vertices:
            return 0.0
        return self.vertexsize / self.plotter.dpi


def register_plotter_scene_objects() -> None:
    """
    Register the force density scene objects to the Plotter context.
    """
    register(FDNetwork, FDNetworkPlotterObject, context="Plotter")
    register(FDMesh, FDMeshPlotterObject, context="Plotter")
