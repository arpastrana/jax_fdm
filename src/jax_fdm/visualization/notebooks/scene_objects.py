from typing import Any

import pythreejs as three
from compas_notebook.scene import ThreeMeshObject
from compas_notebook.scene import ThreeSceneObject

from compas.colors import Color
from compas.scene import register
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization.buffers import Soup
from jax_fdm.visualization.buffers import arrows_buffer
from jax_fdm.visualization.buffers import cylinders_buffer
from jax_fdm.visualization.buffers import soup_colors_rgb
from jax_fdm.visualization.buffers import spheres_buffer
from jax_fdm.visualization.style import ARROW_BODYWIDTH
from jax_fdm.visualization.style import ARROW_HEADPORTION
from jax_fdm.visualization.style import ARROW_HEADWIDTH
from jax_fdm.visualization.style import COLOR_LOAD
from jax_fdm.visualization.style import LOAD_SCALE
from jax_fdm.visualization.style import LOAD_TOL
from jax_fdm.visualization.style import REACTION_SCALE
from jax_fdm.visualization.style import REACTION_TOL
from jax_fdm.visualization.style import edge_colors
from jax_fdm.visualization.style import edge_widths
from jax_fdm.visualization.style import load_arrows
from jax_fdm.visualization.style import point_colors
from jax_fdm.visualization.style import point_sizes
from jax_fdm.visualization.style import reaction_arrows
from jax_fdm.visualization.style import reaction_color_default

__all__ = ["ThreeFDDatastructureObject",
           "ThreeFDNetworkObject",
           "ThreeFDMeshObject",
           "register_notebook_scene_objects"]


class ThreeFDDatastructureObject(ThreeSceneObject):
    """
    A scene object that renders a force density datastructure in a notebook scene.

    Every element category (edges as cylinders, points as spheres, loads and
    reactions as arrows) is batched by the shared soup kernels into a single
    pythreejs mesh with per-vertex colors, so a whole datastructure costs a
    handful of pythreejs objects instead of two per element.

    Notebook rendering is draw-once: unlike the compas_viewer backend, there
    is no scene tree and no in-place update loop for animations.

    Subclasses resolve the point vocabulary (a network addresses its points
    as nodes, a mesh as vertices) by implementing the ``point_*`` methods and
    by exposing it as constructor keyword arguments (``nodecolor``/``nodesize``/
    ``show_nodes`` on a network, ``vertexcolor``/``vertexsize``/``show_vertices``
    on a mesh) that map onto the neutral point parameters here.
    """

    # Tessellation resolution of the batched shapes. Half the viewer default:
    # per-face colors carry the information, not the shading.
    shape_u = 8
    arrow_u = 8

    def __init__(
        self,
        item: FDNetwork | FDMesh | None = None,
        points: list[int] | None = None,
        edges: list[tuple[int, int]] | None = None,
        pointcolor: Color | dict | str | None = None,
        edgecolor: Color | dict | str | None = None,
        pointsize: float | dict | None = None,
        edgewidth: float | dict | tuple | None = None,
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
        self.datastructure: FDNetwork | FDMesh = item  # pyright: ignore[reportAttributeAccessIssue]  # always populated before draw()

        self.points: list[int] = list(points) if points is not None else list(self.point_keys())
        self.edges: list[tuple[int, int]] = list(edges) if edges is not None else list(item.edges())  # pyright: ignore[reportOptionalMemberAccess,reportArgumentType,reportAttributeAccessIssue]  # item is always populated before draw(); edges() with data=False always yields plain (u, v) keys

        # The point-edge adjacency is cached once at construction, so drawing
        # never re-derives it from the datastructure (Mesh.vertex_edges scans
        # all mesh edges per call).
        self.adjacency = {point: list(self.point_edges(point)) for point in self.points}

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

        self.show_points = bool(show_points)
        self.show_edges = show_edges if show_edges is not None else True
        self.show_loads = show_loads if show_loads is not None else True
        self.show_reactions = show_reactions if show_reactions is not None else True

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
    # Draw
    # ==========================================================================

    def draw(self) -> list[Any]:
        """
        Draw the categories of the datastructure as batched pythreejs objects.
        """
        datastructure = self.datastructure

        edge_width = edge_widths(datastructure, self.edges, self.edgewidth_spec)

        guids = []

        if self.show_edges:
            edge_color = edge_colors(datastructure, self.edges, self.edgecolor_spec)

            starts, ends, radii, colors = [], [], [], []
            for edge in self.edges:
                start, end = datastructure.edge_coordinates(edge)
                starts.append(start)
                ends.append(end)
                radii.append(edge_width[edge] / 2.0)
                colors.append(edge_color[edge].rgba)

            guids.append(self.soup_to_mesh(cylinders_buffer(starts, ends, radii, colors, u=self.shape_u)))

        if self.show_points:
            is_support = self.point_is_support if self.show_supports else (lambda key: False)
            point_color = point_colors(self.points, is_support, self.pointcolor_spec)  # pyright: ignore[reportArgumentType]  # point_colors treats a str spec as an unrecognized dict/Color and falls back to default
            point_size = point_sizes(self.points, self.pointsize_spec)

            centers = [self.point_coordinates(point) for point in self.points]
            radii = [point_size[point] / 2.0 for point in self.points]
            colors = [point_color[point].rgba for point in self.points]

            guids.append(self.soup_to_mesh(spheres_buffer(centers, radii, colors, u=self.shape_u, v=self.shape_u)))

        if self.show_loads:
            origins = [self.point_coordinates(point) for point in self.points]
            loads = [self.point_load(point) for point in self.points]
            clearances = [max((edge_width.get(edge, 0.0) for edge in self.adjacency[point]), default=0.0)
                          for point in self.points]

            anchors, vectors = load_arrows(origins, loads, clearances,
                                           self.load_scale, self.load_tol)
            guids.append(self.arrows_to_mesh(anchors, vectors, self.load_color))

        if self.show_reactions:
            points = [point for point in self.points if self.adjacency[point]]
            origins = [self.point_coordinates(point) for point in points]
            reactions = [self.point_reaction(point) for point in points]
            forces = [[datastructure.edge_force(edge) for edge in self.adjacency[point]]
                      for point in points]

            anchors, vectors = reaction_arrows(origins, reactions, forces,
                                               self.reaction_scale, self.reaction_tol)
            guids.append(self.arrows_to_mesh(anchors, vectors, self.reaction_color))

        self._guids = guids

        return self.guids

    def arrows_to_mesh(self, anchors: list[list[float]], vectors: list[list[float]], color: Color) -> three.Mesh:
        """
        Batch one arrow category into a pythreejs mesh.
        """
        colors = [color.rgba] * len(anchors)
        soup = arrows_buffer(anchors, vectors, colors,
                             head_portion=ARROW_HEADPORTION,
                             head_width=ARROW_HEADWIDTH,
                             body_width=ARROW_BODYWIDTH,
                             u=self.arrow_u)

        return self.soup_to_mesh(soup)

    @staticmethod
    def soup_to_mesh(soup: Soup) -> three.Mesh:
        """
        Wrap a triangle soup into a pythreejs mesh with per-vertex colors.
        """
        positions, _ = soup

        geometry = three.BufferGeometry(
            attributes={
                "position": three.BufferAttribute(positions, normalized=False),
                # pythreejs consumes rgb; the soup kernels emit rgba
                "color": three.BufferAttribute(soup_colors_rgb(soup), normalized=False, itemSize=3),
            }
        )
        material = three.MeshBasicMaterial(side="DoubleSide", vertexColors="VertexColors")

        return three.Mesh(geometry, material)


class ThreeFDNetworkObject(ThreeFDDatastructureObject):
    """
    A scene object that renders a force density network in a notebook scene.

    The network points are styled with the ``nodecolor``, ``nodesize`` and
    ``show_nodes`` keyword arguments, matching the datastructure vocabulary.
    """

    # Narrows the base class's FDNetwork | FDMesh attribute to the type this
    # subclass actually holds, so the network-vocabulary accessors below
    # type-check against the right datastructure.
    datastructure: FDNetwork

    def __init__(self, item: FDNetwork | None = None, nodecolor: Color | dict | str | None = None,
                 nodesize: float | dict | None = None, show_nodes: bool | None = None, **kwargs: Any) -> None:
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

    def point_keys(self) -> list[int]:
        return self.datastructure.nodes()  # pyright: ignore[reportReturnType]  # data=False getter always yields plain node keys

    def point_coordinates(self, key: int) -> list[float]:
        return self.datastructure.node_coordinates(key)

    def point_load(self, key: int) -> list[float]:
        return self.datastructure.node_load(key)  # pyright: ignore[reportReturnType]  # getter-mode call always returns a list

    def point_reaction(self, key: int) -> list[float]:
        return self.datastructure.node_reaction(key)

    def point_edges(self, key: int) -> list[tuple[int, int]]:
        return self.datastructure.node_edges(key)

    def point_is_support(self, key: int) -> bool:
        return self.datastructure.is_node_support(key)


class ThreeFDMeshObject(ThreeFDDatastructureObject):
    """
    A scene object that renders a force density mesh in a notebook scene.

    On top of the shared edge/vertex/load/reaction categories, the mesh faces
    are drawn as one shaded surface (the mesh itself).

    The mesh points are styled with the ``vertexcolor``, ``vertexsize`` and
    ``show_vertices`` keyword arguments, matching the datastructure vocabulary.

    The compas_viewer backend's ``faceopacity`` has no notebook counterpart:
    the pythreejs materials compas_notebook builds do not support opacity.
    """

    # Narrows the base class's FDNetwork | FDMesh attribute to the type this
    # subclass actually holds, so the vertex-vocabulary accessors below
    # type-check against the right datastructure.
    datastructure: FDMesh

    def __init__(self,
                 item: FDMesh | None = None,
                 vertexcolor: Color | dict | str | None = None,
                 vertexsize: float | dict | None = None,
                 show_vertices: bool | None = None,
                 show_faces: bool = True,
                 **kwargs: Any) -> None:
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
        self.show_faces = show_faces if show_faces is not None else True

    def point_keys(self) -> list[int]:
        return self.datastructure.vertices()  # pyright: ignore[reportReturnType]  # data=False getter always yields plain vertex keys

    def point_coordinates(self, key: int) -> list[float]:
        return self.datastructure.vertex_coordinates(key)  # pyright: ignore[reportReturnType]  # getter-mode call always returns a list

    def point_load(self, key: int) -> list[float]:
        return self.datastructure.vertex_load(key)  # pyright: ignore[reportReturnType]  # getter-mode call always returns a list

    def point_reaction(self, key: int) -> list[float]:
        return self.datastructure.vertex_reaction(key)

    def point_edges(self, key: int) -> list[tuple[int, int]]:
        return self.datastructure.vertex_edges(key)

    def point_is_support(self, key: int) -> bool:
        return self.datastructure.is_vertex_support(key)

    def draw(self) -> list[Any]:
        """
        Draw the categories of the mesh as batched pythreejs objects.
        """
        guids = super().draw()

        if self.show_faces:
            # sceneobject_type pins the native mesh scene object: the FDMesh is
            # registered with compas.scene, so an unpinned construction would
            # dispatch right back to this class and recurse.
            obj = ThreeMeshObject(item=self.datastructure,
                                  sceneobject_type=ThreeMeshObject,
                                  context="Notebook",
                                  show_edges=True,
                                  show_vertices=False)
            guids += obj.draw()
            self._guids = guids

        return self.guids


def register_notebook_scene_objects():
    """
    Register the force density scene objects to the Notebook context.
    """
    register(FDNetwork, ThreeFDNetworkObject, context="Notebook")
    register(FDMesh, ThreeFDMeshObject, context="Notebook")
