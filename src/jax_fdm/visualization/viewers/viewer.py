from compas_viewer import Viewer as CompasViewer
from compas_viewer.config import Config
from compas_viewer.config import RendererConfig
from compas_viewer.config import WindowConfig

from compas.datastructures import Graph
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization.viewers.buffermanager import FastBufferManager
from jax_fdm.visualization.viewers.mesh_artist import FDMeshViewerArtist
from jax_fdm.visualization.viewers.network_artist import FDNetworkViewerArtist

__all__ = ["Viewer"]


class Viewer(CompasViewer):
    """
    A thin wrapper on :class:`compas_viewer.Viewer`.

    It subclasses the COMPAS viewer so that camera control, the ``on`` frame
    loop, ``show`` and recording all work natively, and only overrides ``add``
    to route a :class:`jax_fdm.datastructures.FDNetwork` through an
    :class:`FDNetworkViewerArtist`.

    For convenience it also accepts the ``width``, ``height`` and ``show_grid``
    keyword arguments directly and folds them into a :class:`compas_viewer.config.Config`,
    so the common window setup does not require building a config by hand.

    The viewer is a process-wide singleton (a compas_viewer constraint):
    a second ``Viewer()`` call returns the same instance with its constructor
    arguments ignored. To visualize several results in one process — e.g. one
    per step of a sequential optimization — reuse the instance::

        viewer = Viewer()
        for step in steps:
            viewer.clear()
            viewer.add(...)
            viewer.show()

    The window closes between shows while the camera carries over.
    """
    def __init__(self, width=None, height=None, show_grid=None, config=None, **kwargs):
        if config is None:
            window = WindowConfig(width=width or 1280, height=height or 720)
            renderer = RendererConfig(show_grid=show_grid if show_grid is not None else True,
                                      rendermode="lighted")
            config = Config(window=window, renderer=renderer)

        super().__init__(config=config, **kwargs)
        self.artists = []
        # Swap in the vectorized buffer manager before any GL buffers exist.
        self.renderer.buffer_manager = FastBufferManager()

    def clear(self):
        """
        Empty the scene for the next round of adds.

        Call between sequential shows, while the window is closed. Besides the
        scene objects this also drops the artists and the picking-color
        registrations, which the parent scene never prunes.
        """
        self.scene.clear()
        self.scene.instance_colors.clear()
        self.artists = []

    def show(self):
        """
        Show the viewer window and block until it is closed.

        The window can be shown again after closing: a re-show rebuilds the GL
        render buffers for the current scene, and ``running`` is reset on
        return so that between-show ``add`` calls stay lightweight (no per-add
        buffer and sidebar rebuild).
        """
        # On a re-show the GL context already exists (the first show created
        # the render buffers via initializeGL, which does not run again), so
        # the buffers must be rebuilt here for the repopulated scene.
        if getattr(self.renderer, "_vao", None) is not None:
            self.renderer.makeCurrent()
            try:
                self.renderer.rebuild_buffers()
            finally:
                self.renderer.doneCurrent()

        try:
            super().show()
        finally:
            self.running = False

    def add(self, data, **kwargs):
        """
        Add a data object to the viewer.

        This is a convenience shortcut for ``viewer.scene.add`` that additionally
        routes a :class:`jax_fdm.datastructures.FDNetwork` through an
        :class:`FDNetworkViewerArtist` and a :class:`jax_fdm.datastructures.FDMesh`
        through an :class:`FDMeshViewerArtist`. ``compas_viewer.Viewer`` itself has
        no ``add`` method (objects go through ``viewer.scene``), so this wrapper
        provides the terser ``viewer.add(obj)`` interface.

        Dispatch is purely by type. To draw a force-density datastructure as plain
        geometry instead (a bare wireframe or shaded surface), convert it to its
        COMPAS base first and add that, e.g. ``viewer.add(mesh.copy(cls=Mesh))``
        or ``viewer.add(network.copy(cls=Network))``.

        Parameters
        ----------
        data : :class:`compas.data.Data`
            The object to visualize.
        **kwargs : dict, optional
            Additional visualization options.

        Returns
        -------
        The created scene object, or the viewer artist for an ``FDNetwork`` /
        ``FDMesh``.
        """
        if isinstance(data, FDMesh):
            return self._add_artist(FDMeshViewerArtist, data, **kwargs)

        if isinstance(data, FDNetwork):
            return self._add_artist(FDNetworkViewerArtist, data, **kwargs)

        # COMPAS 2.x aliases ``Network`` to ``Graph``, so a plain network added as
        # a reference wireframe would otherwise show up as "Graph" in the scene
        # tree. Default its display name to "Network" to avoid surprising users.
        if isinstance(data, Graph) and "name" not in kwargs:
            kwargs["name"] = "Network"

        # compas_viewer's line width kwarg is ``linewidth`` (screen-space pixels);
        # the ``edgewidth`` it inherits from compas.scene is stored but never
        # consumed by the render pipeline. Alias it so ``edgewidth`` is the one
        # edge-width vocabulary across FD and plain adds alike.
        if "edgewidth" in kwargs and "linewidth" not in kwargs:
            kwargs["linewidth"] = kwargs.pop("edgewidth")

        return self.scene.add(data, **kwargs)

    def _add_artist(self, artist_cls, data, **kwargs):
        """
        Build a force-density viewer artist, draw it and add it to the scene.
        """
        artist = artist_cls(data, viewer=self, **kwargs)
        self.artists.append(artist)
        artist.draw()
        artist.add()

        return artist
