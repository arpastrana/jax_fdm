from compas_notebook.scene import ThreeMeshObject

from jax_fdm.visualization.artists import FDMeshArtist
from jax_fdm.visualization.notebooks.datastructure_artist import FDDatastructureNotebookArtist

__all__ = ["FDMeshNotebookArtist"]


class FDMeshNotebookArtist(FDDatastructureNotebookArtist, FDMeshArtist):
    """
    An artist that draws a force density mesh to a :class:`compas_notebook.viewer.Viewer`.

    It reuses the batched notebook machinery of :class:`FDDatastructureNotebookArtist`
    (edges as cylinders, vertices as spheres, load and reaction arrows) paired
    with the mesh's ``vertex_*`` vocabulary via :class:`FDMeshArtist`, and draws
    the mesh faces as the mesh itself.

    The compas_viewer backend's ``faceopacity`` has no notebook counterpart:
    the pythreejs materials compas_notebook builds do not support opacity.
    """

    def draw_guids(self):
        """
        Draw the collections of the mesh and batch them into pythreejs objects.

        On top of the shared edge/vertex/load/reaction categories, the mesh
        faces are drawn as one shaded surface (the mesh itself).
        """
        guids = super().draw_guids()

        if self.show_faces:
            guids += self.draw_faces_guids()

        return guids

    def draw_faces_guids(self):
        """
        Draw the faces of the mesh as pythreejs objects.
        """
        # sceneobject_type pins the native mesh scene object: the FDMesh is
        # registered with compas.scene, so a plain dispatch would route right
        # back to the FD adapter and recurse.
        obj = ThreeMeshObject(item=self.datastructure,
                              sceneobject_type=ThreeMeshObject,
                              context="Notebook",
                              show_edges=True,
                              show_vertices=False)

        return obj.draw()
