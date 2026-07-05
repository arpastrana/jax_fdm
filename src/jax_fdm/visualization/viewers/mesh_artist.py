from jax_fdm.visualization.artists import FDMeshArtist
from jax_fdm.visualization.viewers.datastructure_artist import FDDatastructureViewerArtist

__all__ = ["FDMeshViewerArtist"]


class FDMeshViewerArtist(FDDatastructureViewerArtist, FDMeshArtist):
    """
    An artist that draws a force density mesh to a :class:`compas_viewer.Viewer`.

    It reuses all the scene machinery of :class:`FDDatastructureViewerArtist`
    (edges as cylinders, vertices as spheres, load and reaction arrows) paired
    with the mesh's ``vertex_*`` vocabulary via :class:`FDMeshArtist`, and adds
    the mesh faces as a shaded surface under a "Faces" subgroup.
    """
    points_group_name = "Vertices"

    default_faceopacity = 0.4

    def __init__(self, mesh, viewer, *args, faceopacity=None, **kwargs):
        super().__init__(mesh, viewer, *args, **kwargs)
        self.face_opacity = faceopacity or self.default_faceopacity
        self.viewer_faces = None

    def add(self):
        """
        Add the points of the mesh to the viewer scene.

        On top of the shared edge/vertex/load/reaction groups, the mesh faces are
        drawn as one shaded surface (the mesh itself) under a "Faces" subgroup, so
        the surface toggles independently from the wireframe.
        """
        super().add()

        if self.show_faces:
            self.viewer_groups["faces"] = self.viewer.scene.add_group(name="Faces", parent=self.viewer_group)
            self.viewer_faces = self.viewer.scene.add(self.datastructure,
                                                      show_points=False,
                                                      show_lines=False,
                                                      opacity=self.face_opacity,
                                                      parent=self.viewer_groups["faces"],
                                                      name="Faces")
