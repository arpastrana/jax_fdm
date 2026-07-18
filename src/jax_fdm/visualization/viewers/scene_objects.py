from typing import Any

from compas_viewer.scene import MeshObject
from compas_viewer.scene import ViewerSceneObject

from compas.colors import Color
from compas.scene import register
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization.buffers import FacesData
from jax_fdm.visualization.buffers import Soup
from jax_fdm.visualization.buffers import arrows_buffer
from jax_fdm.visualization.buffers import cylinders_buffer
from jax_fdm.visualization.buffers import soup_indices
from jax_fdm.visualization.buffers import spheres_buffer
from jax_fdm.visualization.style import ARROW_BODYWIDTH
from jax_fdm.visualization.style import ARROW_HEADPORTION
from jax_fdm.visualization.style import ARROW_HEADWIDTH
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
from jax_fdm.visualization.style import load_arrows as style_load_arrows
from jax_fdm.visualization.style import point_colors
from jax_fdm.visualization.style import point_sizes
from jax_fdm.visualization.style import reaction_arrows as style_reaction_arrows
from jax_fdm.visualization.style import reaction_color_default

__all__ = [
    "FDDatastructureObject",
    "FDNetworkObject",
    "FDMeshObject",
    "FDGroupObject",
    "FDObject",
    "register_viewer_scene_objects",
]

# An rgba color, as compas Color.rgba yields it.
RGBA = tuple[float, float, float, float]


# ==========================================================================
# Category children: fused soups
# ==========================================================================


class FDBufferObject(ViewerSceneObject):
    """
    A category child that batches one element category into a triangle soup.

    The soup (edges as cylinders, points as spheres, loads and reactions as
    arrows) is computed on demand from the parent's style state, so the native
    update path (``update(update_data=True)`` re-invoking ``_read_*_data``)
    re-batches against the live datastructure with no extra machinery.

    The soup topology is frozen at add time: arrows collapse to a degenerate
    slot at their anchor while below tolerance and cylinders keep their vertex
    count as widths change, so in-place buffer updates never resize.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "FDBufferObject":
        # Bypass the SceneObject factory: category children wrap no data item,
        # so there is nothing to dispatch on (same pattern as compas Group).
        return object.__new__(cls)

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(
            item=None,
            name=name,
            context="Viewer",
            show_points=False,
            show_lines=False,
            **kwargs,
        )
        self._soup: Soup | None = None

    def build_soup(self) -> Soup:
        """
        Batch the category into (positions, colors) soup arrays.

        Reads the style state of ``self.parent``, the force density scene
        object this category belongs to.
        """
        raise NotImplementedError

    def _read_frontfaces_data(self) -> FacesData:
        self._soup = self.build_soup()
        positions, colors = self._soup
        return positions, colors, soup_indices(self._soup)

    def _read_backfaces_data(self) -> FacesData:
        # The buffer managers always read the front faces first, so the soup
        # computed there is reused with flipped winding.
        soup = self._soup if self._soup is not None else self.build_soup()
        positions, colors = soup
        return positions, colors, soup_indices(soup, flipped=True)


class FDEdgesObject(FDBufferObject):
    """
    The edges of a force density datastructure, batched as cylinders.
    """

    def build_soup(self) -> Soup:
        # a category child is always added under an FDDatastructureObject parent
        parent: FDDatastructureObject = self.parent  # pyright: ignore[reportAssignmentType]

        datastructure = parent.datastructure

        starts, ends, radii, colors = [], [], [], []
        for edge in parent.edges:
            start, end = datastructure.edge_coordinates(edge)
            starts.append(start)
            ends.append(end)
            radii.append(parent.edge_width[edge] / 2.0)
            colors.append(parent.edge_color[edge].rgba)

        return cylinders_buffer(starts, ends, radii, colors, u=parent.shape_u)


class FDPointsObject(FDBufferObject):
    """
    The points (nodes or vertices) of a force density datastructure, batched as spheres.
    """

    def build_soup(self) -> Soup:
        # a category child is always added under an FDDatastructureObject parent
        parent: FDDatastructureObject = self.parent  # pyright: ignore[reportAssignmentType]

        centers, radii, colors = [], [], []
        for point in parent.points:
            centers.append(parent.point_coordinates(point))
            radii.append(parent.point_size[point] / 2.0)
            colors.append(parent.point_color[point].rgba)

        return spheres_buffer(
            centers,
            radii,
            colors,
            u=parent.shape_u,
            v=parent.shape_u,
        )


class FDArrowsObject(FDBufferObject):
    """
    One arrow category (loads or reactions) of a force density datastructure.
    """

    arrows_attr: str | None = None

    def build_soup(self) -> Soup:
        parent = self.parent
        # subclasses always set arrows_attr to a str before instantiation
        anchors, vectors, colors = getattr(parent, self.arrows_attr)()  # pyright: ignore[reportArgumentType]

        return arrows_buffer(
            anchors,
            vectors,
            colors,
            head_portion=ARROW_HEADPORTION,
            head_width=ARROW_HEADWIDTH,
            body_width=ARROW_BODYWIDTH,
            u=parent.arrow_u,
        )


class FDLoadsObject(FDArrowsObject):
    arrows_attr = "load_arrows"


class FDReactionsObject(FDArrowsObject):
    arrows_attr = "reaction_arrows"


# ==========================================================================
# Category children: per-element groups
# ==========================================================================


class FDGroupObject(ViewerSceneObject):
    """
    A non-drawing category node grouping per-element children.

    The group must be a plain scene object rather than a compas Group: the
    render buffers skip Group instances, which would sever the settings-buffer
    parent chain the shader walks to inherit show, selection and transform
    from the force density parent down to every element child.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "FDGroupObject":
        # Bypass the SceneObject factory: the group wraps no data item,
        # so there is nothing to dispatch on (same pattern as compas Group).
        return object.__new__(cls)

    def __init__(self, name: str, **kwargs: Any) -> None:
        super().__init__(
            item=None,
            name=name,
            context="Viewer",
            show_points=False,
            show_lines=False,
            **kwargs,
        )

    def update(self, update_transform: bool = True, update_data: bool = True) -> None:
        for child in self.children:
            child.update(update_transform=update_transform, update_data=update_data)


class FDObject(FDBufferObject):
    """
    A single element (edge, point, load or reaction arrow) of a force density
    datastructure, as its own selectable scene object.

    The element holds only its key: style and geometry are read from the
    force density parent two levels up, and the soup is built by the same
    buffer builders as the fused categories with a single-element batch, so
    both render paths are vertex-identical by construction.
    """

    def __init__(self, key: int | tuple[int, int], name: str, **kwargs: Any) -> None:
        super().__init__(name=name, **kwargs)
        self.key = key

    @property
    def fd_parent(self) -> "FDDatastructureObject":
        # The element sits under a category group under the FD parent.
        return self.parent.parent


class FDEdgeObject(FDObject):
    """
    One edge of a force density datastructure, as a cylinder.
    """

    # Narrows the base class's int | tuple[int, int] key to the edge key
    # this subclass actually holds.
    key: tuple[int, int]

    def build_soup(self) -> Soup:
        parent = self.fd_parent
        start, end = parent.datastructure.edge_coordinates(self.key)

        # edge_coordinates always returns plain xyz lists, not compas attribute
        # views.
        return cylinders_buffer(
            [start],  # pyright: ignore[reportArgumentType]
            [end],  # pyright: ignore[reportArgumentType]
            [parent.edge_width[self.key] / 2.0],
            [parent.edge_color[self.key].rgba],
            u=parent.shape_u,
        )


class FDPointObject(FDObject):
    """
    One point (node or vertex) of a force density datastructure, as a sphere.
    """

    # Narrows the base class's int | tuple[int, int] key to the point key
    # this subclass actually holds.
    key: int

    def build_soup(self) -> Soup:
        parent = self.fd_parent

        return spheres_buffer(
            [parent.point_coordinates(self.key)],
            [parent.point_size[self.key] / 2.0],
            [parent.point_color[self.key].rgba],
            u=parent.shape_u,
            v=parent.shape_u,
        )


class FDArrowObject(FDObject):
    """
    One arrow (load or reaction) of a force density datastructure.
    """

    arrow_attr: str | None = None

    # Narrows the base class's int | tuple[int, int] key to the point key
    # this subclass actually holds.
    key: int

    def build_soup(self) -> Soup:
        parent = self.fd_parent
        # subclasses always set arrow_attr to a str before instantiation
        anchor, vector, color = getattr(parent, self.arrow_attr)(self.key)  # pyright: ignore[reportArgumentType]

        return arrows_buffer(
            [anchor],
            [vector],
            [color],
            head_portion=ARROW_HEADPORTION,
            head_width=ARROW_HEADWIDTH,
            body_width=ARROW_BODYWIDTH,
            u=parent.arrow_u,
        )


class FDLoadObject(FDArrowObject):
    arrow_attr = "load_arrow"


class FDReactionObject(FDArrowObject):
    arrow_attr = "reaction_arrow"


# ==========================================================================
# Parent scene objects
# ==========================================================================


class FDDatastructureObject(ViewerSceneObject):
    """
    A scene object that renders a force density datastructure in a compas_viewer scene.

    By default every element (edges as cylinders, points as spheres, loads
    and reactions as arrows) is its own scene object under a per-category
    group, so single elements are clickable, highlightable and foldable in
    the viewer tree. Arrows below tolerance at add time are pruned rather
    than kept as invisible slots.

    With ``fuse=True`` every category batches into one child scene object
    holding a single triangle soup instead, so a whole datastructure costs a
    handful of scene objects and an animation loop updates one render buffer
    per category in place via ``update()``. Both render paths build their
    soups with the same buffer builders, so they are vertex-identical.

    The parent itself draws nothing: it owns the datastructure, the style
    state (computed at construction and re-derived on every ``update``) and
    the frozen candidate lists of the arrow categories.

    Styling and topology are frozen at add time: the force density keyword
    arguments are constructor parameters, the point-edge adjacency is cached
    once, and restyling or reconnecting means re-adding the object.

    Subclasses resolve the point vocabulary (a network addresses its points
    as nodes, a mesh as vertices) by implementing the ``point_*`` methods and
    by exposing it as constructor keyword arguments (``nodes``/``nodecolor``/
    ``nodesize``/``show_nodes`` on a network, ``vertices``/``vertexcolor``/
    ``vertexsize``/``show_vertices`` on a mesh) that map onto the neutral
    point parameters here.
    """

    points_name = "Points"
    point_name = "Point"

    default_opacity = 0.75

    shape_u = 16
    arrow_u = 8

    FUSE_HINT_ELEMENTS = 1000

    def __init__(
        self,
        item: FDNetwork | FDMesh | None = None,
        points: list[int] | None = None,
        edges: list[tuple[int, int]] | None = None,
        pointcolor: PointColorSpec = None,
        edgecolor: EdgeColorSpec = None,
        pointsize: PointSizeSpec = None,
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
        fuse: bool = False,
        **kwargs: Any,
    ) -> None:
        # The pin kwarg used to bypass registry dispatch is not a scene kwarg.
        kwargs.pop("sceneobject_type", None)
        super().__init__(item=item, **kwargs)

        # The scene registry always dispatches a real datastructure to this
        # constructor; the Optional in the signature only matches the base
        # class default.
        self.datastructure: FDNetwork | FDMesh = item

        # Point and edge iterables, optionally filtered (defaults to all).
        self.points: list[int] = (
            list(points) if points is not None else list(self.point_keys())
        )
        # item is always populated before draw(); edges() with data=False yields
        # plain (u, v) keys.
        self.edges: list[tuple[int, int]] = (
            list(edges) if edges is not None else list(item.edges())  # pyright: ignore[reportOptionalMemberAccess]
        )

        # Connectivity is frozen at add time, like the soup topology: the
        # point-edge adjacency is cached once so per-frame updates never
        # re-derive it from the datastructure (Mesh.vertex_edges scans all
        # mesh edges per call).
        self.adjacency = {point: list(self.point_edges(point)) for point in self.points}

        # Style inputs, kept raw: semantic modes ("force", "fd", (min, max))
        # are re-derived against the live datastructure on every update. The
        # spec suffix keeps them apart from the computed per-element dicts
        # (edge_color et al.) and from the plain floats the base class owns
        # (pointsize et al.).
        self.pointcolor_spec = pointcolor
        self.edgecolor_spec = edgecolor
        self.pointsize_spec = pointsize
        self.edgewidth_spec = edgewidth
        self.show_supports = show_supports if show_supports is not None else True

        self.load_color = loadcolor or COLOR_LOAD
        self.load_scale = loadscale or LOAD_SCALE
        self.load_tol = loadtol or LOAD_TOL

        self.reaction_color = reactioncolor or reaction_color_default(edgecolor)
        self.reaction_scale = reactionscale or REACTION_SCALE
        self.reaction_tol = reactiontol or REACTION_TOL

        # Populated by recompute() below, before any draw() reads them.
        self.edge_color: dict[tuple[int, int], Color]
        self.edge_width: dict[tuple[int, int], float]
        self.point_color: dict[int, Color]
        self.point_size: dict[int, float]
        self.recompute()

        # Candidate point lists of the arrow categories, frozen so the soup
        # membership never changes across updates.
        self.load_points = list(self.points)
        self.reaction_points = [point for point in self.points if self.adjacency[point]]

        # One child per shown category: a fused soup, or a group of
        # per-element children. Scene backends may inject explicit None
        # values for the show flags, which mean "default".
        self.fuse = fuse
        if show_edges or show_edges is None:
            self._add_category(FDEdgesObject, FDEdgeObject, "Edges", self.edges, "Edge")
        if show_points:
            self._add_category(
                FDPointsObject,
                FDPointObject,
                self.points_name,
                self.points,
                self.point_name,
            )
        if show_reactions or show_reactions is None:
            self._add_category(
                FDReactionsObject,
                FDReactionObject,
                "Reactions",
                self._arrow_points("reaction_arrow", self.reaction_points),
                "Reaction",
            )
        if show_loads or show_loads is None:
            self._add_category(
                FDLoadsObject,
                FDLoadObject,
                "Loads",
                self._arrow_points("load_arrow", self.load_points),
                "Load",
            )

        if not fuse:
            count = sum(
                len(child.children)
                for child in self.children
                if isinstance(child, FDGroupObject)
            )
            if count > self.FUSE_HINT_ELEMENTS:
                print(
                    f"WARNING: {self.name} has {count} per-element scene objects. "
                    "Pass fuse=True to viewer.add(...) for fast loading and display",
                )

    def _add_category(
        self,
        fused_cls: type,
        element_cls: type,
        category_name: str,
        keys: list[int] | list[tuple[int, int]],
        element_name: str,
    ) -> None:
        """
        Add one category child: a fused soup, or a group of per-element children.

        Element children spawn in fused soup order and carry the opacity
        themselves: the shader inherits show, selection and transform through
        the parent chain, but not opacity.
        """
        if self.fuse:
            self.add(fused_cls(name=category_name, opacity=self.default_opacity))
            return

        group = FDGroupObject(name=category_name)
        self.add(group)
        for key in keys:
            group.add(
                element_cls(
                    key,
                    name=f"{element_name} {key}",
                    opacity=self.default_opacity,
                ),
            )

    def _arrow_points(self, arrow_attr: str, points: list[int]) -> list[int]:
        """
        The candidate points whose arrow is visible at add time.

        The fused soups keep a degenerate slot for every candidate so their
        topology never changes; per-element children prune instead, so an
        arrow appearing after add means re-adding the object.
        """
        if self.fuse:
            return points
        arrow = getattr(self, arrow_attr)
        return [point for point in points if any(arrow(point)[1])]

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

    # ==========================================================================
    # Style state
    # ==========================================================================

    def recompute(self) -> None:
        """
        Derive the per-element style state from the live datastructure.

        Widths must precede the arrows: the load arrow anchor backs off by the
        width of the thickest connected edge.
        """
        datastructure = self.datastructure

        self.edge_color = edge_colors(datastructure, self.edges, self.edgecolor_spec)
        self.edge_width = edge_widths(datastructure, self.edges, self.edgewidth_spec)

        is_support = (
            self.point_is_support if self.show_supports else (lambda key: False)
        )
        # point_colors treats a str spec as an unrecognized dict/Color and falls
        # back to default
        self.point_color = point_colors(self.points, is_support, self.pointcolor_spec)  # pyright: ignore[reportArgumentType]
        self.point_size = point_sizes(self.points, self.pointsize_spec)

    def load_arrows(self) -> tuple[list[list[float]], list[list[float]], list[RGBA]]:
        """
        The anchors, vectors and colors of the load arrows.
        """
        points = self.load_points
        origins = [self.point_coordinates(point) for point in points]
        loads = [self.point_load(point) for point in points]
        clearances = [self._point_clearance(point) for point in points]

        anchors, vectors = style_load_arrows(
            origins,
            loads,
            clearances,
            self.load_scale,
            self.load_tol,
        )

        return anchors, vectors, self._arrow_colors(points, self.load_color)

    def reaction_arrows(
        self,
    ) -> tuple[list[list[float]], list[list[float]], list[RGBA]]:
        """
        The anchors, vectors and colors of the reaction arrows.
        """
        points = self.reaction_points
        origins = [self.point_coordinates(point) for point in points]
        reactions = [self.point_reaction(point) for point in points]
        forces = [
            [self.datastructure.edge_force(edge) for edge in self.adjacency[point]]
            for point in points
        ]

        anchors, vectors = style_reaction_arrows(
            origins,
            reactions,
            forces,
            self.reaction_scale,
            self.reaction_tol,
        )

        return anchors, vectors, self._arrow_colors(points, self.reaction_color)

    def load_arrow(self, point: int) -> tuple[list[float], list[float], RGBA]:
        """
        The anchor, vector and color of the load arrow at one point.
        """
        anchors, vectors = style_load_arrows(
            [self.point_coordinates(point)],
            [self.point_load(point)],
            [self._point_clearance(point)],
            self.load_scale,
            self.load_tol,
        )

        return anchors[0], vectors[0], self._arrow_colors([point], self.load_color)[0]

    def reaction_arrow(self, point: int) -> tuple[list[float], list[float], RGBA]:
        """
        The anchor, vector and color of the reaction arrow at one point.
        """
        forces = [self.datastructure.edge_force(edge) for edge in self.adjacency[point]]
        anchors, vectors = style_reaction_arrows(
            [self.point_coordinates(point)],
            [self.point_reaction(point)],
            [forces],
            self.reaction_scale,
            self.reaction_tol,
        )

        return (
            anchors[0],
            vectors[0],
            self._arrow_colors([point], self.reaction_color)[0],
        )

    def _point_clearance(self, point: int) -> float:
        """
        The width of the thickest edge connected to a point.
        """
        return max(
            (self.edge_width.get(edge, 0.0) for edge in self.adjacency[point]),
            default=0.0,
        )

    @staticmethod
    def _arrow_colors(points: list[int], color: Color | dict[int, Color]) -> list[RGBA]:
        if isinstance(color, dict):
            return [color[point].rgba for point in points]
        return [color.rgba] * len(points)

    # ==========================================================================
    # Update
    # ==========================================================================

    def update(self, update_transform: bool = True, update_data: bool = True) -> None:
        """
        Update the render buffers of the datastructure in place.

        Call after mutating the datastructure (e.g. per animation frame):
        the style state is re-derived and every category child re-batches
        its soup against the live geometry.
        """
        self.recompute()

        for child in self.children:
            child.update(update_transform=update_transform, update_data=update_data)


class FDNetworkObject(FDDatastructureObject):
    """
    A scene object that renders a force density network in a compas_viewer scene.

    The network points are filtered with the ``nodes`` keyword argument and
    styled with ``nodecolor``, ``nodesize`` and ``show_nodes``, matching the
    datastructure vocabulary.
    """

    points_name = "Nodes"
    point_name = "Node"

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
        # base. The scene backend injects the neutral names with explicit None
        # values (meaning "default"), so they are popped and only kept when
        # no node keyword takes precedence.
        points = kwargs.pop("points", None)
        pointcolor = kwargs.pop("pointcolor", None)
        pointsize = kwargs.pop("pointsize", None)
        show_points = kwargs.pop("show_points", None)
        super().__init__(
            item=item,
            points=nodes if nodes is not None else points,
            pointcolor=nodecolor if nodecolor is not None else pointcolor,
            pointsize=nodesize if nodesize is not None else pointsize,
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


class FDMeshObject(FDDatastructureObject):
    """
    A scene object that renders a force density mesh in a compas_viewer scene.

    On top of the shared edge/vertex/load/reaction categories, the mesh faces
    are drawn as one shaded surface (the mesh itself), so the surface toggles
    independently from the wireframe.

    The mesh points are filtered with the ``vertices`` keyword argument and
    styled with ``vertexcolor``, ``vertexsize`` and ``show_vertices``,
    matching the datastructure vocabulary.
    """

    points_name = "Vertices"
    point_name = "Vertex"

    default_faceopacity = 0.4

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
        faceopacity: float | None = None,
        show_faces: bool = True,
        **kwargs: Any,
    ) -> None:
        # Map the vertex vocabulary onto the neutral point parameters of the
        # base. The scene backend injects the neutral names with explicit None
        # values (meaning "default"), so they are popped and only kept when
        # no vertex keyword takes precedence.
        points = kwargs.pop("points", None)
        pointcolor = kwargs.pop("pointcolor", None)
        pointsize = kwargs.pop("pointsize", None)
        show_points = kwargs.pop("show_points", None)
        super().__init__(
            item=item,
            points=vertices if vertices is not None else points,
            pointcolor=vertexcolor if vertexcolor is not None else pointcolor,
            pointsize=vertexsize if vertexsize is not None else pointsize,
            show_points=show_vertices if show_vertices is not None else show_points,
            **kwargs,
        )

        if show_faces or show_faces is None:
            # sceneobject_type pins the native mesh scene object: the FDMesh is
            # registered with compas.scene, so an unpinned construction would
            # dispatch right back to this class and recurse.
            faces = MeshObject(
                item=item,
                sceneobject_type=MeshObject,
                context="Viewer",
                name="Faces",
                show_points=False,
                show_lines=False,
                opacity=faceopacity or self.default_faceopacity,
            )
            self.add(faces)

    def point_keys(self) -> list[int]:
        # the data=False getter always yields plain vertex keys
        return self.datastructure.vertices()  # pyright: ignore[reportReturnType]

    def point_coordinates(self, key: int) -> list[float]:
        # the getter-mode call always returns a list
        return self.datastructure.vertex_coordinates(key)  # pyright: ignore[reportReturnType]

    def point_load(self, key: int) -> list[float]:
        # the getter-mode call always returns a list
        return self.datastructure.vertex_load(key)  # pyright: ignore[reportReturnType]

    def point_reaction(self, key: int) -> list[float]:
        return self.datastructure.vertex_reaction(key)

    def point_edges(self, key: int) -> list[tuple[int, int]]:
        return self.datastructure.vertex_edges(key)

    def point_is_support(self, key: int) -> bool:
        return self.datastructure.is_vertex_support(key)


# ==========================================================================
# Registration
# ==========================================================================


def register_viewer_scene_objects() -> None:
    """
    Register the force density scene objects to the Viewer context.
    """
    register(FDNetwork, FDNetworkObject, context="Viewer")
    register(FDMesh, FDMeshObject, context="Viewer")
