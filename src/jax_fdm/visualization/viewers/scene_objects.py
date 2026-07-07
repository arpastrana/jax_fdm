import numpy as np
from compas_viewer.scene import MeshObject
from compas_viewer.scene import ViewerSceneObject

from compas.scene import register
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization import scene as fdscene
from jax_fdm.visualization.buffers import arrows_buffer
from jax_fdm.visualization.buffers import cylinders_buffer
from jax_fdm.visualization.buffers import spheres_buffer

__all__ = ["FDDatastructureObject",
           "FDNetworkObject",
           "FDMeshObject",
           "register_viewer_scene_objects"]


# ==========================================================================
# Category children
# ==========================================================================

class _FDBufferObject(ViewerSceneObject):
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
        return positions, colors, np.arange(len(positions))

    def _read_backfaces_data(self):
        # The buffer managers always read the front faces first, so the soup
        # computed there is reused with flipped winding.
        positions, colors = self._soup if self._soup is not None else self._build_soup()
        return positions, colors, np.flip(np.arange(len(positions)))


class _FDEdgesObject(_FDBufferObject):
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

        return cylinders_buffer(starts, ends, radii, np.asarray(colors), u=parent.shape_u)


class _FDPointsObject(_FDBufferObject):
    """
    The points (nodes or vertices) of a force density datastructure, batched as spheres.
    """

    def _build_soup(self):
        parent = self.parent

        centers, radii, colors = [], [], []
        for point in parent.points:
            centers.append(parent.access.coordinates(point))
            radii.append(parent.point_size[point] / 2.0)
            colors.append(parent.point_color[point].rgba)

        return spheres_buffer(centers, radii, np.asarray(colors), u=parent.shape_u, v=parent.shape_u)


class _FDArrowsObject(_FDBufferObject):
    """
    One arrow category (loads or reactions) of a force density datastructure.
    """
    arrows_attr = None

    def _build_soup(self):
        parent = self.parent
        anchors, vectors, colors = getattr(parent, self.arrows_attr)()

        return arrows_buffer(anchors, vectors, colors,
                             head_portion=fdscene.ARROW_HEADPORTION,
                             head_width=fdscene.ARROW_HEADWIDTH,
                             body_width=fdscene.ARROW_BODYWIDTH,
                             u=parent.arrow_u)


class _FDLoadsObject(_FDArrowsObject):
    arrows_attr = "load_arrows"


class _FDReactionsObject(_FDArrowsObject):
    arrows_attr = "reaction_arrows"


# ==========================================================================
# Parent scene objects
# ==========================================================================

class FDDatastructureObject(ViewerSceneObject):
    """
    A scene object that renders a force density datastructure in a compas_viewer scene.

    Every element category (edges as cylinders, points as spheres, loads and
    reactions as arrows) is one child scene object holding a single batched
    triangle soup, so a whole datastructure costs a handful of scene objects,
    each individually toggleable in the viewer tree, and an animation loop
    updates one render buffer per category in place via ``update()``.

    The parent itself draws nothing: it owns the datastructure, the style
    state (computed at construction and re-derived on every ``update``) and
    the frozen candidate lists of the arrow categories.

    Styling is frozen at add time: the force density keyword arguments are
    constructor parameters, and restyling means re-adding the object.
    """
    accessors = None
    points_name = "Points"

    default_opacity = 0.75

    shape_u = 16
    arrow_u = 8

    def __init__(self,
                 item=None,
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
                 **kwargs):
        # The pin kwarg used to bypass registry dispatch is not a scene kwarg.
        kwargs.pop("sceneobject_type", None)
        super().__init__(item=item, **kwargs)

        self.datastructure = item
        self.access = self.accessors(item)

        # Point and edge iterables, optionally filtered (defaults to all).
        self.points = list(points) if points is not None else list(self.access.keys())
        self.edges = list(edges) if edges is not None else list(item.edges())

        # Style inputs, kept raw: semantic modes ("force", "fd", (min, max))
        # are re-derived against the live datastructure on every update.
        self._nodecolor = nodecolor
        self._edgecolor = edgecolor
        self._nodesize = nodesize
        self._edgewidth = edgewidth
        self._show_supports = show_supports if show_supports is not None else True

        self.load_color = loadcolor or fdscene.COLOR_LOAD
        self.load_scale = loadscale or fdscene.LOAD_SCALE
        self.load_tol = loadtol or fdscene.LOAD_TOL

        self.reaction_color = reactioncolor or fdscene.reaction_color_default(edgecolor)
        self.reaction_scale = reactionscale or fdscene.REACTION_SCALE
        self.reaction_tol = reactiontol or fdscene.REACTION_TOL

        self.edge_color = None
        self.edge_width = None
        self.point_color = None
        self.point_size = None
        self._recompute()

        # Candidate point lists of the arrow categories, frozen so the soup
        # membership never changes across updates.
        self._load_points = list(self.points)
        self._reaction_points = [point for point in self.points if list(self.access.edges(point))]

        # One child scene object per shown category. Scene backends may inject
        # explicit None values for the show flags, which mean "default".
        opacity = self.default_opacity
        if show_edges or show_edges is None:
            self.add(_FDEdgesObject(name="Edges", opacity=opacity))
        if show_nodes:
            self.add(_FDPointsObject(name=self.points_name, opacity=opacity))
        if show_reactions or show_reactions is None:
            self.add(_FDReactionsObject(name="Reactions", opacity=opacity))
        if show_loads or show_loads is None:
            self.add(_FDLoadsObject(name="Loads", opacity=opacity))

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

        self.edge_color = fdscene.edge_colors(datastructure, self.edges, self._edgecolor)
        self.edge_width = fdscene.edge_widths(datastructure, self.edges, self._edgewidth)

        is_support = self.access.is_support if self._show_supports else (lambda key: False)
        self.point_color = fdscene.point_colors(self.points, is_support, self._nodecolor)
        self.point_size = fdscene.point_sizes(self.points, self._nodesize)

    def load_arrows(self):
        """
        The anchors, vectors and colors of the load arrows.
        """
        anchors, vectors = fdscene.load_arrows(self._load_points,
                                               self.access,
                                               self.edge_width,
                                               self.load_scale,
                                               self.load_tol)
        colors = self._arrow_colors(self._load_points, self.load_color)

        return anchors, vectors, colors

    def reaction_arrows(self):
        """
        The anchors, vectors and colors of the reaction arrows.
        """
        anchors, vectors = fdscene.reaction_arrows(self._reaction_points,
                                                   self.access,
                                                   self.datastructure,
                                                   self.reaction_scale,
                                                   self.reaction_tol)
        colors = self._arrow_colors(self._reaction_points, self.reaction_color)

        return anchors, vectors, colors

    @staticmethod
    def _arrow_colors(points, color):
        if isinstance(color, dict):
            return np.asarray([color[point].rgba for point in points])
        return np.asarray([color.rgba] * len(points))

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
    """
    accessors = staticmethod(fdscene.network_accessors)
    points_name = "Nodes"


class FDMeshObject(FDDatastructureObject):
    """
    A scene object that renders a force density mesh in a compas_viewer scene.

    On top of the shared edge/vertex/load/reaction categories, the mesh faces
    are drawn as one shaded surface (the mesh itself), so the surface toggles
    independently from the wireframe.
    """
    accessors = staticmethod(fdscene.mesh_accessors)
    points_name = "Vertices"

    default_faceopacity = 0.4

    def __init__(self, item=None, faceopacity=None, show_faces=True, **kwargs):
        super().__init__(item=item, **kwargs)

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


# ==========================================================================
# Registration
# ==========================================================================

def register_viewer_scene_objects():
    """
    Register the force density scene objects to the Viewer context.
    """
    register(FDNetwork, FDNetworkObject, context="Viewer")
    register(FDMesh, FDMeshObject, context="Viewer")
