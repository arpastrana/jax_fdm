import numpy as np
import pythreejs as three
from compas_notebook.scene import ThreeMeshObject
from compas_notebook.scene import ThreeSceneObject

from compas.scene import register
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization import scene as fdscene
from jax_fdm.visualization.buffers import arrows_buffer
from jax_fdm.visualization.buffers import cylinders_buffer
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
    """
    accessors = None

    # Tessellation resolution of the batched shapes. Half the viewer default:
    # per-face colors carry the information, not the shading.
    shape_u = 8
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
        kwargs.pop("sceneobject_type", None)
        super().__init__(item=item, **kwargs)

        self.datastructure = item
        self.access = self.accessors(item)

        self.points = list(points) if points is not None else list(self.access.keys())
        self.edges = list(edges) if edges is not None else list(item.edges())

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

        self.show_nodes = show_nodes
        self.show_edges = show_edges if show_edges is not None else True
        self.show_loads = show_loads if show_loads is not None else True
        self.show_reactions = show_reactions if show_reactions is not None else True

    # ==========================================================================
    # Draw
    # ==========================================================================

    def draw(self):
        """
        Draw the categories of the datastructure as batched pythreejs objects.
        """
        datastructure = self.datastructure

        edge_width = fdscene.edge_widths(datastructure, self.edges, self._edgewidth)

        guids = []

        if self.show_edges:
            edge_color = fdscene.edge_colors(datastructure, self.edges, self._edgecolor)

            starts, ends, radii, colors = [], [], [], []
            for edge in self.edges:
                start, end = datastructure.edge_coordinates(edge)
                starts.append(start)
                ends.append(end)
                radii.append(edge_width[edge] / 2.0)
                colors.append(edge_color[edge].rgba)

            guids.append(self.soup_to_mesh(cylinders_buffer(starts, ends, radii, np.asarray(colors), u=self.shape_u)))

        if self.show_nodes:
            is_support = self.access.is_support if self._show_supports else (lambda key: False)
            point_color = fdscene.point_colors(self.points, is_support, self._nodecolor)
            point_size = fdscene.point_sizes(self.points, self._nodesize)

            centers = [self.access.coordinates(point) for point in self.points]
            radii = [point_size[point] / 2.0 for point in self.points]
            colors = np.asarray([point_color[point].rgba for point in self.points])

            guids.append(self.soup_to_mesh(spheres_buffer(centers, radii, colors, u=self.shape_u, v=self.shape_u)))

        if self.show_loads:
            anchors, vectors = fdscene.load_arrows(self.points, self.access, edge_width,
                                                   self.load_scale, self.load_tol)
            guids.append(self.arrows_to_mesh(anchors, vectors, self.load_color))

        if self.show_reactions:
            points = [point for point in self.points if list(self.access.edges(point))]
            anchors, vectors = fdscene.reaction_arrows(points, self.access, datastructure,
                                                       self.reaction_scale, self.reaction_tol)
            guids.append(self.arrows_to_mesh(anchors, vectors, self.reaction_color))

        self._guids = guids

        return self.guids

    def arrows_to_mesh(self, anchors, vectors, color):
        """
        Batch one arrow category into a pythreejs mesh.
        """
        colors = np.asarray([color.rgba] * len(anchors))
        soup = arrows_buffer(anchors, vectors, colors,
                             head_portion=fdscene.ARROW_HEADPORTION,
                             head_width=fdscene.ARROW_HEADWIDTH,
                             body_width=fdscene.ARROW_BODYWIDTH,
                             u=self.arrow_u)

        return self.soup_to_mesh(soup)

    @staticmethod
    def soup_to_mesh(soup):
        """
        Wrap a triangle soup into a pythreejs mesh with per-vertex colors.
        """
        positions, colors = soup

        geometry = three.BufferGeometry(
            attributes={
                "position": three.BufferAttribute(positions, normalized=False),
                # pythreejs consumes rgb; the soup kernels emit rgba
                "color": three.BufferAttribute(np.ascontiguousarray(colors[:, :3]), normalized=False, itemSize=3),
            }
        )
        material = three.MeshBasicMaterial(side="DoubleSide", vertexColors="VertexColors")

        return three.Mesh(geometry, material)


class ThreeFDNetworkObject(ThreeFDDatastructureObject):
    """
    A scene object that renders a force density network in a notebook scene.
    """
    accessors = staticmethod(fdscene.network_accessors)


class ThreeFDMeshObject(ThreeFDDatastructureObject):
    """
    A scene object that renders a force density mesh in a notebook scene.

    On top of the shared edge/vertex/load/reaction categories, the mesh faces
    are drawn as one shaded surface (the mesh itself).

    The compas_viewer backend's ``faceopacity`` has no notebook counterpart:
    the pythreejs materials compas_notebook builds do not support opacity.
    """
    accessors = staticmethod(fdscene.mesh_accessors)

    def __init__(self, item=None, show_faces=True, **kwargs):
        super().__init__(item=item, **kwargs)
        self.show_faces = show_faces if show_faces is not None else True

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
