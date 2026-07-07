from compas_viewer import Viewer as CompasViewer
from compas_viewer.config import Config
from compas_viewer.config import RendererConfig
from compas_viewer.config import WindowConfig
from compas_viewer.singleton import Singleton
from compas_viewer.singleton import SingletonMeta
from PySide6.QtWidgets import QApplication

from compas.datastructures import Graph
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization.viewers.buffermanager import FastBufferManager
from jax_fdm.visualization.viewers.mesh_artist import FDMeshViewerArtist
from jax_fdm.visualization.viewers.network_artist import FDNetworkViewerArtist

__all__ = ["Viewer"]


def retire_viewer(viewer):
    """
    Best-effort shutdown of a spent viewer.

    Its timers must stop so they do not resume firing into a dead GL context
    when the shared ``QApplication`` enters the event loop of the next viewer.
    Stopping suffices: a retired timer is orphaned and never restarted.
    """
    timers = [getattr(viewer, "timer", None)]
    try:
        timers.append(viewer.renderer._idle_timer)
    except AttributeError:
        pass

    for timer in timers:
        try:
            timer.stop()
        except (AttributeError, RuntimeError):
            pass

    try:
        viewer.ui.window.widget.close()
    except (AttributeError, RuntimeError):
        pass

    try:
        viewer.running = False
    except AttributeError:
        pass


class ViewerMeta(SingletonMeta):
    """
    A singleton metaclass that evicts a spent viewer.

    ``compas_viewer.Viewer`` is a process-wide singleton whose ``running``
    flag is set by ``show()`` and never reset when the window closes. A second
    ``Viewer()`` in the same process then returns the closed instance, still
    flagged as running and still holding the previous scene, so every
    ``scene.add`` triggers a full buffer and sidebar rebuild against a dead GL
    context (quadratic slowdown, or a crash).

    This metaclass retires such a spent (or foreign, non-jax_fdm) cached
    instance before construction, so that each sequential ``Viewer()`` starts
    with a fresh scene, UI and renderer.
    """
    def __call__(cls, *args, **kwargs):
        # SingletonMeta caches instances under the first subclass of Singleton
        # in the MRO, which for this chain is compas_viewer.Viewer.
        key_class = cls
        while key_class.__base__ is not Singleton:
            key_class = key_class.__base__

        cached = SingletonMeta._instances.get(key_class)
        if cached is not None:
            spent = getattr(cached, "running", False) or getattr(cached, "_spent", False)
            if spent or not isinstance(cached, cls):
                retire_viewer(cached)
                del SingletonMeta._instances[key_class]

        return super().__call__(*args, **kwargs)


class Viewer(CompasViewer, metaclass=ViewerMeta):
    """
    A thin wrapper on :class:`compas_viewer.Viewer`.

    It subclasses the COMPAS viewer so that camera control, the ``on`` frame
    loop, ``show`` and recording all work natively, and only overrides ``add``
    to route a :class:`jax_fdm.datastructures.FDNetwork` through an
    :class:`FDNetworkViewerArtist`.

    For convenience it also accepts the ``width``, ``height`` and ``show_grid``
    keyword arguments directly and folds them into a :class:`compas_viewer.config.Config`,
    so the common window setup does not require building a config by hand.

    Unlike the parent singleton, this viewer is safe to construct, show and
    close several times in one process (e.g. one viewer per step of a
    sequential optimization): a spent instance is evicted by :class:`ViewerMeta`
    and the next construction starts from a clean scene.
    """
    def __init__(self, width=None, height=None, show_grid=None, config=None, **kwargs):
        if config is None:
            window = WindowConfig(width=width or 1280, height=height or 720)
            renderer = RendererConfig(show_grid=show_grid if show_grid is not None else True,
                                      rendermode="lighted")
            config = Config(window=window, renderer=renderer)

        super().__init__(config=config, **kwargs)
        self.artists = []
        self._spent = False
        # Swap in the vectorized buffer manager before any GL buffers exist.
        self.renderer.buffer_manager = FastBufferManager()

    def create_app(self):
        """
        Reuse the process-wide Qt application if one exists.

        A fresh instance built after an eviction must not create a second
        ``QApplication`` — that is a fatal Qt error.
        """
        app = QApplication.instance()
        if app is not None:
            return app
        return super().create_app()

    def show(self):
        """
        Show the viewer window and block until it is closed.

        On return the instance is marked as spent: late ``scene.add`` calls
        skip the live-rebuild path, and the next ``Viewer()`` constructs fresh.
        """
        super().show()
        self.running = False
        self._spent = True

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
