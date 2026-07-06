from compas_notebook.scene import ThreeSceneObject

from compas.scene import register
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization.artists.datastructure_artist import pop_artist_kwargs
from jax_fdm.visualization.notebooks.mesh_artist import FDMeshNotebookArtist
from jax_fdm.visualization.notebooks.network_artist import FDNetworkNotebookArtist

__all__ = ["ThreeFDDatastructureObject",
           "ThreeFDNetworkObject",
           "ThreeFDMeshObject",
           "register_notebook_scene_objects"]


class ThreeFDDatastructureObject(ThreeSceneObject):
    """
    A scene object that renders a force density datastructure in a native notebook scene.

    This is an interop adapter: it lets a plain :class:`compas.scene.Scene` (or a
    bare compas_notebook viewer) display a force density datastructure exactly as
    the jax_fdm artists dictate, with all the render logic staying in the artist.
    It draws flat (a list of batched pythreejs objects) because the Notebook
    context has no working scene groups.
    """
    artist_cls = None

    def __init__(self, item=None, artist=None, **kwargs):
        kwargs.pop("sceneobject_type", None)
        artist_kwargs = pop_artist_kwargs(kwargs)
        super().__init__(item=item, **kwargs)
        self.artist = artist or self.artist_cls(item, **artist_kwargs)

    def draw(self):
        """
        Draw the datastructure through its notebook artist.
        """
        self._guids = self.artist.draw_guids()
        return self.guids


class ThreeFDNetworkObject(ThreeFDDatastructureObject):
    """
    A scene object that renders a force density network in a native notebook scene.
    """
    artist_cls = FDNetworkNotebookArtist


class ThreeFDMeshObject(ThreeFDDatastructureObject):
    """
    A scene object that renders a force density mesh in a native notebook scene.
    """
    artist_cls = FDMeshNotebookArtist


def register_notebook_scene_objects():
    """
    Register the force density scene objects to the Notebook context.
    """
    register(FDNetwork, ThreeFDNetworkObject, context="Notebook")
    register(FDMesh, ThreeFDMeshObject, context="Notebook")
