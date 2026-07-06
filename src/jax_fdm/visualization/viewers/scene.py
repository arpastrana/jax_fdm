from compas_viewer.scene import Group

from compas.scene import register
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization.artists.datastructure_artist import pop_artist_kwargs
from jax_fdm.visualization.viewers.mesh_artist import FDMeshViewerArtist
from jax_fdm.visualization.viewers.network_artist import FDNetworkViewerArtist

__all__ = ["FDDatastructureViewerObject",
           "FDNetworkViewerObject",
           "FDMeshViewerObject",
           "register_viewer_scene_objects"]


class _GroupScene:
    """
    A minimal viewer-scene stand-in that routes the artist's adds into a group subtree.

    ``SceneObject.add`` needs no tree membership, so the whole subtree can be
    built inside the scene object's ``__init__``, before ``Scene.add`` parents
    it into the scene tree; the renderer then initializes every descendant
    (at show time, or through ``rebuild_buffers`` for runtime adds).
    """
    def __init__(self, root):
        self.root = root

    def add(self, item, parent=None, **kwargs):
        return (parent or self.root).add(item, **kwargs)

    def add_group(self, name=None, parent=None):
        group = Group(name=name)
        (parent or self.root).add(group)
        return group


class _GroupHost:
    """
    A duck-typed viewer host for the artists, which only ever touch ``viewer.scene``.
    """
    def __init__(self, root):
        self.scene = _GroupScene(root)


class FDDatastructureViewerObject(Group):
    """
    A scene object that renders a force density datastructure in a native viewer scene.

    This is an interop adapter: it lets a bare :class:`compas_viewer.Viewer`
    display a force density datastructure exactly as the jax_fdm artists dictate,
    with all the render logic staying in the artist. The adapter is a scene group
    under which the artist parents its per-category subgroups, so the foldable
    tree matches the one built by the jax_fdm viewer wrapper.
    """
    artist_cls = None

    def __init__(self, item=None, name=None, **kwargs):
        kwargs.pop("sceneobject_type", None)
        artist_kwargs = pop_artist_kwargs(kwargs)
        # A group requires a name; SceneObject.__new__ only passes the item.
        name = name or getattr(item, "name", None) or type(item).__name__
        super().__init__(item=item, name=name, **kwargs)
        # Group kwargs cascade to every child added under it and would collide
        # on "context" and "item" in Scene.add; the artist sets every child
        # kwarg explicitly, so inherit nothing.
        self.kwargs.clear()

        self.artist = self.artist_cls(item, viewer=_GroupHost(self), **artist_kwargs)
        self.artist.draw()
        self.artist.add(group=self)


class FDNetworkViewerObject(FDDatastructureViewerObject):
    """
    A scene object that renders a force density network in a native viewer scene.
    """
    artist_cls = FDNetworkViewerArtist


class FDMeshViewerObject(FDDatastructureViewerObject):
    """
    A scene object that renders a force density mesh in a native viewer scene.
    """
    artist_cls = FDMeshViewerArtist


def register_viewer_scene_objects():
    """
    Register the force density scene objects to the Viewer context.
    """
    register(FDNetwork, FDNetworkViewerObject, context="Viewer")
    register(FDMesh, FDMeshViewerObject, context="Viewer")
