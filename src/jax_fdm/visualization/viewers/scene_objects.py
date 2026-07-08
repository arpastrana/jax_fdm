from compas_viewer.scene import MeshObject
from compas_viewer.scene import ViewerSceneObject

from compas.scene import register
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization import style
from jax_fdm.visualization.buffers import arrows_buffer
from jax_fdm.visualization.buffers import cylinders_buffer
from jax_fdm.visualization.buffers import soup_indices
from jax_fdm.visualization.buffers import spheres_buffer

__all__ = ["FDDatastructureObject",
           "FDNetworkObject",
           "FDMeshObject",
           "FDGroupObject",
           "FDObject",
           "register_viewer_scene_objects"]


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

    def __new__(cls, *args, **kwargs):
        # Bypass the SceneObject factory: category children wrap no data item,
        # so there is nothing to dispatch on (same pattern as compas Group).
        return object.__new__(cls)

    def __init__(self, name, **kwargs):
        super().__init__(item=None, name=name, context="Viewer",
                         show_points=False, show_lines=False, **kwargs)
        self._soup = None

    def _build_soup(self):
        """
        Batch the category into (positions, colors) soup arrays.

        Reads the style state of ``self.parent``, the force density scene
        object this category belongs to.
        """
        raise NotImplementedError

    def _read_frontfaces_data(self):
        self._soup = self._build_soup()
        positions, colors = self._soup
        return positions, colors, soup_indices(self._soup)

    def _read_backfaces_data(self):
        # The buffer managers always read the front faces first, so the soup
        # computed there is reused with flipped winding.
        soup = self._soup if self._soup is not None else self._build_soup()
        positions, colors = soup
        return positions, colors, soup_indices(soup, flipped=True)


class FDEdgesObject(FDBufferObject):
    """
    The edges of a force density datastructure, batched as cylinders.
    """

    def _build_soup(self):
        parent = self.parent
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

    def _build_soup(self):
        parent = self.parent

        centers, radii, colors = [], [], []
        for point in parent.points:
            centers.append(parent.point_coordinates(point))
            radii.append(parent.point_size[point] / 2.0)
            colors.append(parent.point_color[point].rgba)

        return spheres_buffer(centers, radii, colors, u=parent.shape_u, v=parent.shape_u)


class FDArrowsObject(FDBufferObject):
    """
    One arrow category (loads or reactions) of a force density datastructure.
    """
    arrows_attr = None

    def _build_soup(self):
        parent = self.parent
        anchors, vectors, colors = getattr(parent, self.arrows_attr)()

        return arrows_buffer(anchors, vectors, colors,
                             head_portion=style.ARROW_HEADPORTION,
                             head_width=style.ARROW_HEADWIDTH,
                             body_width=style.ARROW_BODYWIDTH,
                             u=parent.arrow_u)


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

    def __new__(cls, *args, **kwargs):
        # Bypass the SceneObject factory: the group wraps no data item,
        # so there is nothing to dispatch on (same pattern as compas Group).
        return object.__new__(cls)

    def __init__(self, name, **kwargs):
        super().__init__(item=None, name=name, context="Viewer",
                         show_points=False, show_lines=False, **kwargs)

    def update(self, update_transform=True, update_data=True):
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

    def __init__(self, key, name, **kwargs):
        super().__init__(name=name, **kwargs)
        self.key = key

    @property
    def fdparent(self):
        # The element sits under a category group under the FD parent.
        return self.parent.parent


class FDEdgeObject(FDObject):
    """
    One edge of a force density datastructure, as a cylinder.
    """

    def _build_soup(self):
        parent = self.fdparent
        start, end = parent.datastructure.edge_coordinates(self.key)

        return cylinders_buffer([start], [end],
                                [parent.edge_width[self.key] / 2.0],
                                [parent.edge_color[self.key].rgba],
                                u=parent.shape_u)


class FDPointObject(FDObject):
    """
    One point (node or vertex) of a force density datastructure, as a sphere.
    """

    def _build_soup(self):
        parent = self.fdparent

        return spheres_buffer([parent.point_coordinates(self.key)],
                              [parent.point_size[self.key] / 2.0],
                              [parent.point_color[self.key].rgba],
                              u=parent.shape_u, v=parent.shape_u)


class FDArrowObject(FDObject):
    """
    One arrow (load or reaction) of a force density datastructure.
    """
    arrow_attr = None

    def _build_soup(self):
        parent = self.fdparent
        anchor, vector, color = getattr(parent, self.arrow_attr)(self.key)

        return arrows_buffer([anchor], [vector], [color],
                             head_portion=style.ARROW_HEADPORTION,
                             head_width=style.ARROW_HEADWIDTH,
                             body_width=style.ARROW_BODYWIDTH,
                             u=parent.arrow_u)


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
    by exposing it as constructor keyword arguments (``nodecolor``/``nodesize``/
    ``show_nodes`` on a network, ``vertexcolor``/``vertexsize``/``show_vertices``
    on a mesh) that map onto the neutral point parameters here.
    """
    points_name = "Points"
    point_name = "Point"

    default_opacity = 0.75

    shape_u = 16
    arrow_u = 8

    FUSE_HINT_ELEMENTS = 1000

    def __init__(self,
                 item=None,
                 points=None,
                 edges=None,
                 pointcolor=None,
                 edgecolor=None,
                 pointsize=None,
                 edgewidth=None,
                 loadcolor=None,
                 loadscale=None,
                 loadtol=None,
                 reactioncolor=None,
                 reactionscale=None,
                 reactiontol=None,
                 show_points=False,
                 show_edges=True,
                 show_loads=True,
                 show_reactions=True,
                 show_supports=True,
                 fuse=False,
                 **kwargs):
        # The pin kwarg used to bypass registry dispatch is not a scene kwarg.
        kwargs.pop("sceneobject_type", None)
        super().__init__(item=item, **kwargs)

        self.datastructure = item

        # Point and edge iterables, optionally filtered (defaults to all).
        self.points = list(points) if points is not None else list(self.point_keys())
        self.edges = list(edges) if edges is not None else list(item.edges())

        # Connectivity is frozen at add time, like the soup topology: the
        # point-edge adjacency is cached once so per-frame updates never
        # re-derive it from the datastructure (Mesh.vertex_edges scans all
        # mesh edges per call).
        self._adjacency = {point: list(self.point_edges(point)) for point in self.points}

        # Style inputs, kept raw: semantic modes ("force", "fd", (min, max))
        # are re-derived against the live datastructure on every update.
        self._pointcolor = pointcolor
        self._edgecolor = edgecolor
        self._pointsize = pointsize
        self._edgewidth = edgewidth
        self._show_supports = show_supports if show_supports is not None else True

        self.load_color = loadcolor or style.COLOR_LOAD
        self.load_scale = loadscale or style.LOAD_SCALE
        self.load_tol = loadtol or style.LOAD_TOL

        self.reaction_color = reactioncolor or style.reaction_color_default(edgecolor)
        self.reaction_scale = reactionscale or style.REACTION_SCALE
        self.reaction_tol = reactiontol or style.REACTION_TOL

        self.edge_color = None
        self.edge_width = None
        self.point_color = None
        self.point_size = None
        self._recompute()

        # Candidate point lists of the arrow categories, frozen so the soup
        # membership never changes across updates.
        self._load_points = list(self.points)
        self._reaction_points = [point for point in self.points if self._adjacency[point]]

        # One child per shown category: a fused soup, or a group of
        # per-element children. Scene backends may inject explicit None
        # values for the show flags, which mean "default".
        self.fuse = fuse
        if show_edges or show_edges is None:
            self._add_category(FDEdgesObject, FDEdgeObject, "Edges",
                               self.edges, "Edge")
        if show_points:
            self._add_category(FDPointsObject, FDPointObject, self.points_name,
                               self.points, self.point_name)
        if show_reactions or show_reactions is None:
            self._add_category(FDReactionsObject, FDReactionObject, "Reactions",
                               self._arrow_points("reaction_arrow", self._reaction_points), "Reaction")
        if show_loads or show_loads is None:
            self._add_category(FDLoadsObject, FDLoadObject, "Loads",
                               self._arrow_points("load_arrow", self._load_points), "Load")

        if not fuse:
            count = sum(len(child.children) for child in self.children
                        if isinstance(child, FDGroupObject))
            if count > self.FUSE_HINT_ELEMENTS:
                print(f"WARNING: {self.name} has {count} per-element scene objects. "
                      "Pass fuse=True to viewer.add(...) for fast loading and display")

    def _add_category(self, fused_cls, element_cls, category_name, keys, element_name):
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
            group.add(element_cls(key, name=f"{element_name} {key}",
                                  opacity=self.default_opacity))

    def _arrow_points(self, arrow_attr, points):
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

    def point_keys(self):
        raise NotImplementedError

    def point_coordinates(self, key):
        raise NotImplementedError

    def point_load(self, key):
        raise NotImplementedError

    def point_reaction(self, key):
        raise NotImplementedError

    def point_edges(self, key):
        raise NotImplementedError

    def point_is_support(self, key):
        raise NotImplementedError

    # ==========================================================================
    # Style state
    # ==========================================================================

    def _recompute(self):
        """
        Derive the per-element style state from the live datastructure.

        Widths must precede the arrows: the load arrow anchor backs off by the
        width of the thickest connected edge.
        """
        datastructure = self.datastructure

        self.edge_color = style.edge_colors(datastructure, self.edges, self._edgecolor)
        self.edge_width = style.edge_widths(datastructure, self.edges, self._edgewidth)

        is_support = self.point_is_support if self._show_supports else (lambda key: False)
        self.point_color = style.point_colors(self.points, is_support, self._pointcolor)
        self.point_size = style.point_sizes(self.points, self._pointsize)

    def load_arrows(self):
        """
        The anchors, vectors and colors of the load arrows.
        """
        points = self._load_points
        origins = [self.point_coordinates(point) for point in points]
        loads = [self.point_load(point) for point in points]
        clearances = [self._point_clearance(point) for point in points]

        anchors, vectors = style.load_arrows(origins, loads, clearances,
                                             self.load_scale, self.load_tol)

        return anchors, vectors, self._arrow_colors(points, self.load_color)

    def reaction_arrows(self):
        """
        The anchors, vectors and colors of the reaction arrows.
        """
        points = self._reaction_points
        origins = [self.point_coordinates(point) for point in points]
        reactions = [self.point_reaction(point) for point in points]
        forces = [[self.datastructure.edge_force(edge) for edge in self._adjacency[point]]
                  for point in points]

        anchors, vectors = style.reaction_arrows(origins, reactions, forces,
                                                 self.reaction_scale, self.reaction_tol)

        return anchors, vectors, self._arrow_colors(points, self.reaction_color)

    def load_arrow(self, point):
        """
        The anchor, vector and color of the load arrow at one point.
        """
        anchors, vectors = style.load_arrows([self.point_coordinates(point)],
                                             [self.point_load(point)],
                                             [self._point_clearance(point)],
                                             self.load_scale, self.load_tol)

        return anchors[0], vectors[0], self._arrow_colors([point], self.load_color)[0]

    def reaction_arrow(self, point):
        """
        The anchor, vector and color of the reaction arrow at one point.
        """
        forces = [self.datastructure.edge_force(edge) for edge in self._adjacency[point]]
        anchors, vectors = style.reaction_arrows([self.point_coordinates(point)],
                                                 [self.point_reaction(point)],
                                                 [forces],
                                                 self.reaction_scale, self.reaction_tol)

        return anchors[0], vectors[0], self._arrow_colors([point], self.reaction_color)[0]

    def _point_clearance(self, point):
        """
        The width of the thickest edge connected to a point.
        """
        return max((self.edge_width.get(edge, 0.0) for edge in self._adjacency[point]), default=0.0)

    @staticmethod
    def _arrow_colors(points, color):
        if isinstance(color, dict):
            return [color[point].rgba for point in points]
        return [color.rgba] * len(points)

    # ==========================================================================
    # Update
    # ==========================================================================

    def update(self, update_transform=True, update_data=True):
        """
        Update the render buffers of the datastructure in place.

        Call after mutating the datastructure (e.g. per animation frame):
        the style state is re-derived and every category child re-batches
        its soup against the live geometry.
        """
        self._recompute()

        for child in self.children:
            child.update(update_transform=update_transform, update_data=update_data)


class FDNetworkObject(FDDatastructureObject):
    """
    A scene object that renders a force density network in a compas_viewer scene.

    The network points are styled with the ``nodecolor``, ``nodesize`` and
    ``show_nodes`` keyword arguments, matching the datastructure vocabulary.
    """
    points_name = "Nodes"
    point_name = "Node"

    def __init__(self, item=None, nodecolor=None, nodesize=None, show_nodes=None, **kwargs):
        # Map the node vocabulary onto the neutral point parameters of the
        # base. The scene backend injects the neutral names with explicit None
        # values (meaning "default"), so they are popped and only kept when
        # no node keyword takes precedence.
        pointcolor = kwargs.pop("pointcolor", None)
        pointsize = kwargs.pop("pointsize", None)
        show_points = kwargs.pop("show_points", None)
        super().__init__(item=item,
                         pointcolor=nodecolor if nodecolor is not None else pointcolor,
                         pointsize=nodesize if nodesize is not None else pointsize,
                         show_points=show_nodes if show_nodes is not None else show_points,
                         **kwargs)

    def point_keys(self):
        return self.datastructure.nodes()

    def point_coordinates(self, key):
        return self.datastructure.node_coordinates(key)

    def point_load(self, key):
        return self.datastructure.node_load(key)

    def point_reaction(self, key):
        return self.datastructure.node_reaction(key)

    def point_edges(self, key):
        return self.datastructure.node_edges(key)

    def point_is_support(self, key):
        return self.datastructure.is_node_support(key)


class FDMeshObject(FDDatastructureObject):
    """
    A scene object that renders a force density mesh in a compas_viewer scene.

    On top of the shared edge/vertex/load/reaction categories, the mesh faces
    are drawn as one shaded surface (the mesh itself), so the surface toggles
    independently from the wireframe.

    The mesh points are styled with the ``vertexcolor``, ``vertexsize`` and
    ``show_vertices`` keyword arguments, matching the datastructure vocabulary.
    """
    points_name = "Vertices"
    point_name = "Vertex"

    default_faceopacity = 0.4

    def __init__(self,
                 item=None,
                 vertexcolor=None,
                 vertexsize=None,
                 show_vertices=None,
                 faceopacity=None,
                 show_faces=True,
                 **kwargs):
        # Map the vertex vocabulary onto the neutral point parameters of the
        # base. The scene backend injects the neutral names with explicit None
        # values (meaning "default"), so they are popped and only kept when
        # no vertex keyword takes precedence.
        pointcolor = kwargs.pop("pointcolor", None)
        pointsize = kwargs.pop("pointsize", None)
        show_points = kwargs.pop("show_points", None)
        super().__init__(item=item,
                         pointcolor=vertexcolor if vertexcolor is not None else pointcolor,
                         pointsize=vertexsize if vertexsize is not None else pointsize,
                         show_points=show_vertices if show_vertices is not None else show_points,
                         **kwargs)

        if show_faces or show_faces is None:
            # sceneobject_type pins the native mesh scene object: the FDMesh is
            # registered with compas.scene, so an unpinned construction would
            # dispatch right back to this class and recurse.
            faces = MeshObject(item=item,
                               sceneobject_type=MeshObject,
                               context="Viewer",
                               name="Faces",
                               show_points=False,
                               show_lines=False,
                               opacity=faceopacity or self.default_faceopacity)
            self.add(faces)

    def point_keys(self):
        return self.datastructure.vertices()

    def point_coordinates(self, key):
        return self.datastructure.vertex_coordinates(key)

    def point_load(self, key):
        return self.datastructure.vertex_load(key)

    def point_reaction(self, key):
        return self.datastructure.vertex_reaction(key)

    def point_edges(self, key):
        return self.datastructure.vertex_edges(key)

    def point_is_support(self, key):
        return self.datastructure.is_vertex_support(key)


# ==========================================================================
# Registration
# ==========================================================================

def register_viewer_scene_objects():
    """
    Register the force density scene objects to the Viewer context.
    """
    register(FDNetwork, FDNetworkObject, context="Viewer")
    register(FDMesh, FDMeshObject, context="Viewer")
