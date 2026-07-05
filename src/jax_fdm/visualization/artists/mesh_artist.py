from jax_fdm.visualization.artists.datastructure_artist import FDDatastructureArtist

__all__ = ["FDMeshArtist"]


class FDMeshArtist(FDDatastructureArtist):
    """
    The base artist to display a force density mesh across different contexts.

    It implements the point hooks of :class:`FDDatastructureArtist` in terms of
    the mesh's ``vertex_*`` vocabulary and exposes a ``mesh`` alias. On top of the
    shared edge/load/reaction rendering it adds the faces of the mesh as a shaded
    surface, controlled by ``show_faces``.
    """

    def __init__(self, mesh, *args, show_faces=True, **kwargs):
        super().__init__(mesh, *args, **kwargs)
        self.show_faces = show_faces

    # ==========================================================================
    # Point hooks
    # ==========================================================================

    def _points(self):
        return self.datastructure.vertices()

    def _point_coordinates(self, key):
        return self.datastructure.vertex_coordinates(key)

    def _point_load(self, key):
        return self.datastructure.vertex_load(key)

    def _point_reaction(self, key):
        return self.datastructure.vertex_reaction(key)

    def _point_edges(self, key):
        return self.datastructure.vertex_edges(key)

    def _point_is_support(self, key):
        return self.datastructure.vertex_attribute(key, "is_support")

    def _point_label(self, key):
        return f"Vertex ({key})"

    # ==========================================================================
    # Aliases (mesh vocabulary)
    # ==========================================================================

    @property
    def mesh(self):
        return self.datastructure

    @mesh.setter
    def mesh(self, mesh):
        self.datastructure = mesh

    @property
    def vertices(self):
        return self.points

    @vertices.setter
    def vertices(self, vertices):
        self.points = vertices
