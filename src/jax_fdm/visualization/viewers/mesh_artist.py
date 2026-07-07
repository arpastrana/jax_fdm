from compas_viewer.scene import MeshObject

from jax_fdm.visualization.artists import FDMeshArtist
from jax_fdm.visualization.viewers.datastructure_artist import FDDatastructureViewerArtist

__all__ = ["FDMeshViewerArtist"]


class FDMeshViewerArtist(FDDatastructureViewerArtist, FDMeshArtist):
    """
    An artist that draws a force density mesh to a :class:`compas_viewer.Viewer`.

    It reuses all the batched-buffer machinery of :class:`FDDatastructureViewerArtist`
    (edges as cylinders, vertices as spheres, load and reaction arrows) paired
    with the mesh's ``vertex_*`` vocabulary via :class:`FDMeshArtist`, and adds
    the mesh faces as a shaded surface.
    """
    points_group_name = "Vertices"

    default_faceopacity = 0.4

    def __init__(self, mesh, viewer, *args, faceopacity=None, **kwargs):
        super().__init__(mesh, viewer, *args, **kwargs)
        self.face_opacity = faceopacity or self.default_faceopacity
        self.viewer_faces = None

    def add(self, group=None):
        """
        Add the elements of the mesh to the viewer scene.

        On top of the shared edge/vertex/load/reaction buffers, the mesh faces
        are drawn as one shaded surface (the mesh itself), so the surface
        toggles independently from the wireframe.
        """
        super().add(group=group)

        if not self.show_faces:
            return

        # sceneobject_type pins the native mesh scene object: the FDMesh is
        # registered with compas.scene, so a plain scene.add would dispatch
        # right back to the FD adapter and recurse.
        self.viewer_faces = self.viewer.scene.add(self.datastructure,
                                                  sceneobject_type=MeshObject,
                                                  show_points=False,
                                                  show_lines=False,
                                                  opacity=self.face_opacity,
                                                  parent=self.viewer_group,
                                                  name="Faces")

    def update(self):
        """
        Update the render buffers of the mesh drawn by this artist in place.

        The faces surface re-reads the mesh it wraps, which an animation loop
        mutates in place via ``datastructure_update``.
        """
        super().update()

        if self.viewer_faces is not None:
            self.viewer_faces.update(update_data=True)
