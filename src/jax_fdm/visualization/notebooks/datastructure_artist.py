from compas_notebook.scene import ThreeMeshObject

from compas.datastructures import Mesh
from jax_fdm.visualization.artists import FDShapeArtist

__all__ = ["FDDatastructureNotebookArtist"]


class FDDatastructureNotebookArtist(FDShapeArtist):
    """
    An artist that draws a force density datastructure to a :class:`compas_notebook.viewer.Viewer`.

    The shared draw loop of :class:`FDDatastructureArtist` fills the collections
    with plain COMPAS shapes (cylinders for edges, spheres for points, arrows for
    load and reaction vectors) via :class:`FDShapeArtist`; each category is then
    batched into a single mesh with per-face colors keyed by element, so a whole
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
        Join a collection of shapes into one mesh and draw it with per-face colors.
        """
        mesh, facecolor = self.join_collection(collection, color)

        obj = ThreeMeshObject(item=mesh,
                              sceneobject_type=ThreeMeshObject,
                              context="Notebook",
                              facecolor=facecolor,
                              show_edges=self.show_shape_edges,
                              show_vertices=False)

        return obj.draw()

    def join_collection(self, collection, color):
        """
        Concatenate the shape tessellations of a collection into one mesh.

        Returns the joined mesh and a face-to-color mapping that assigns every
        face the color of the element its shape belongs to.
        """
        vertices = []
        faces = []
        facecolor = {}

        for key, shape in collection.items():
            _color = color[key] if isinstance(color, dict) else color

            shape.resolution_u = self.shape_u
            shape.resolution_v = self.shape_v
            shape_vertices, shape_faces = shape.to_vertices_and_faces()

            offset = len(vertices)
            vertices += [list(xyz) for xyz in shape_vertices]

            for face in shape_faces:
                facecolor[len(faces)] = _color
                faces.append([index + offset for index in face])

        return Mesh.from_vertices_and_faces(vertices, faces), facecolor
