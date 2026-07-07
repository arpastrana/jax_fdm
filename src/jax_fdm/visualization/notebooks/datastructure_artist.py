import numpy as np
import pythreejs as three

from compas.colors import Color
from jax_fdm.visualization.artists import FDShapeArtist

__all__ = ["FDDatastructureNotebookArtist"]


class FDDatastructureNotebookArtist(FDShapeArtist):
    """
    An artist that draws a force density datastructure to a :class:`compas_notebook.viewer.Viewer`.

    The shared draw loop of :class:`FDDatastructureArtist` fills the collections
    with plain COMPAS shapes (cylinders for edges, spheres for points, arrows for
    load and reaction vectors); each category is then batched straight into one
    pythreejs buffer with per-vertex colors keyed by element, so a whole
    datastructure costs a handful of pythreejs objects instead of two per element.

    Notebook rendering is draw-once: unlike the compas_viewer backend, there is
    no scene tree and no in-place update loop for animations.
    """
    # Tessellation resolution of the batched shapes. Half the COMPAS default:
    # per-face colors carry the information, not the shading.
    shape_u = 8
    shape_v = 8

    # Whether to overlay the tessellation wireframe on the batched shapes.
    # pythreejs materials are unlit, so the wireframe adds contour definition
    # at the cost of one extra scene object per category.
    show_shape_edges = False
    shape_edgecolor = Color(0.2, 0.2, 0.2)

    def draw_guids(self):
        """
        Draw the collections of the datastructure and batch them into pythreejs objects.
        """
        self.draw()

        guids = []
        for collection, color in ((self.collection_edges, self.edge_color),
                                  (self.collection_points, self.point_color),
                                  (self.collection_loads, self.load_color),
                                  (self.collection_reactions, self.reaction_color)):
            if collection:
                guids += self.draw_collection(collection, color)

        return guids

    def draw_collection(self, collection, color):
        """
        Batch a collection of shapes into one mesh buffer with per-element colors.
        """
        guids = [self.collection_to_facesbuffer(collection, color)]

        if self.show_shape_edges:
            guids.append(self.collection_to_edgesbuffer(collection))

        return guids

    def collection_to_facesbuffer(self, collection, color):
        """
        Concatenate the triangulated shape tessellations of a collection into one mesh buffer.

        Every triangle takes the color of the element its shape belongs to,
        carried as per-vertex colors.
        """
        positions = []
        colors = []

        for key, shape in collection.items():
            _color = color[key] if isinstance(color, dict) else color

            shape.resolution_u = self.shape_u
            shape.resolution_v = self.shape_v
            vertices, faces = shape.to_vertices_and_faces()

            # Fan-triangulate in place: the shape faces are convex polygons, and
            # ``to_vertices_and_faces(triangulated=True)`` would tessellate the
            # shape a second time (``Shape.vertices`` recomputes on every access).
            for face in faces:
                for i in range(1, len(face) - 1):
                    for index in (face[0], face[i], face[i + 1]):
                        positions.append(vertices[index])
                        colors.append(_color)

        geometry = three.BufferGeometry(
            attributes={
                "position": three.BufferAttribute(np.array(positions, dtype=np.float32), normalized=False),
                "color": three.BufferAttribute(np.array(colors, dtype=np.float32), normalized=False, itemSize=3),
            }
        )
        material = three.MeshBasicMaterial(side="DoubleSide", vertexColors="VertexColors")

        return three.Mesh(geometry, material)

    def collection_to_edgesbuffer(self, collection):
        """
        Concatenate the tessellation wireframes of a collection into one line segment buffer.
        """
        positions = []

        for shape in collection.values():
            shape.resolution_u = self.shape_u
            shape.resolution_v = self.shape_v
            vertices = shape.vertices

            for u, v in shape.edges:
                positions.append(vertices[u])
                positions.append(vertices[v])

        geometry = three.BufferGeometry(
            attributes={
                "position": three.BufferAttribute(np.array(positions, dtype=np.float32), normalized=False),
            }
        )
        material = three.LineBasicMaterial(color=self.shape_edgecolor.hex)

        return three.LineSegments(geometry, material)
