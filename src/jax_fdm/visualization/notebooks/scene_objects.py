import pythreejs as three
from compas_notebook.scene import ThreeMeshObject
from compas_notebook.scene import ThreeSceneObject

from compas.scene import register
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization import style
from jax_fdm.visualization.buffers import arrows_buffer
from jax_fdm.visualization.buffers import cylinders_buffer
from jax_fdm.visualization.buffers import soup_colors_rgb
from jax_fdm.visualization.buffers import spheres_buffer

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
                 **kwargs):
        kwargs.pop("sceneobject_type", None)
        super().__init__(item=item, **kwargs)

        self.datastructure = item

        self.points = list(points) if points is not None else list(self.point_keys())
        self.edges = list(edges) if edges is not None else list(item.edges())

        # The point-edge adjacency is cached once at construction, so drawing
        # never re-derives it from the datastructure (Mesh.vertex_edges scans
        # all mesh edges per call).
        self._adjacency = {point: list(self.point_edges(point)) for point in self.points}

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

        self.show_points = bool(show_points)
        self.show_edges = show_edges if show_edges is not None else True
        self.show_loads = show_loads if show_loads is not None else True
        self.show_reactions = show_reactions if show_reactions is not None else True

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
    # Draw
    # ==========================================================================

    def draw(self):
        """
        Draw the categories of the datastructure as batched pythreejs objects.
        """
        datastructure = self.datastructure

        edge_width = style.edge_widths(datastructure, self.edges, self._edgewidth)

        guids = []

        if self.show_edges:
            edge_color = style.edge_colors(datastructure, self.edges, self._edgecolor)

            starts, ends, radii, colors = [], [], [], []
            for edge in self.edges:
                start, end = datastructure.edge_coordinates(edge)
                starts.append(start)
                ends.append(end)
                radii.append(edge_width[edge] / 2.0)
                colors.append(edge_color[edge].rgba)

            guids.append(self.soup_to_mesh(cylinders_buffer(starts, ends, radii, colors, u=self.shape_u)))

        if self.show_points:
            is_support = self.point_is_support if self._show_supports else (lambda key: False)
            point_color = style.point_colors(self.points, is_support, self._pointcolor)
            point_size = style.point_sizes(self.points, self._pointsize)

            centers = [self.point_coordinates(point) for point in self.points]
            radii = [point_size[point] / 2.0 for point in self.points]
            colors = [point_color[point].rgba for point in self.points]

            guids.append(self.soup_to_mesh(spheres_buffer(centers, radii, colors, u=self.shape_u, v=self.shape_u)))

        if self.show_loads:
            origins = [self.point_coordinates(point) for point in self.points]
            loads = [self.point_load(point) for point in self.points]
            clearances = [max((edge_width.get(edge, 0.0) for edge in self._adjacency[point]), default=0.0)
                          for point in self.points]

            anchors, vectors = style.load_arrows(origins, loads, clearances,
                                                 self.load_scale, self.load_tol)
            guids.append(self.arrows_to_mesh(anchors, vectors, self.load_color))

        if self.show_reactions:
            points = [point for point in self.points if self._adjacency[point]]
            origins = [self.point_coordinates(point) for point in points]
            reactions = [self.point_reaction(point) for point in points]
            forces = [[datastructure.edge_force(edge) for edge in self._adjacency[point]]
                      for point in points]

            anchors, vectors = style.reaction_arrows(origins, reactions, forces,
                                                     self.reaction_scale, self.reaction_tol)
            guids.append(self.arrows_to_mesh(anchors, vectors, self.reaction_color))

        self._guids = guids

        return self.guids

    def arrows_to_mesh(self, anchors, vectors, color):
        """
        Batch one arrow category into a pythreejs mesh.
        """
        colors = [color.rgba] * len(anchors)
        soup = arrows_buffer(anchors, vectors, colors,
                             head_portion=style.ARROW_HEADPORTION,
                             head_width=style.ARROW_HEADWIDTH,
                             body_width=style.ARROW_BODYWIDTH,
                             u=self.arrow_u)

        return self.soup_to_mesh(soup)

    @staticmethod
    def soup_to_mesh(soup):
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

    def __init__(self,
                 item=None,
                 vertexcolor=None,
                 vertexsize=None,
                 show_vertices=None,
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
        self.show_faces = show_faces if show_faces is not None else True

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

    def draw(self):
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
