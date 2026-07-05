from compas_viewer import Viewer as CompasViewer
from compas_viewer.config import Config
from compas_viewer.config import RendererConfig
from compas_viewer.config import WindowConfig

from jax_fdm.datastructures import FDNetwork
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
    """
    def __init__(self, width=None, height=None, show_grid=None, config=None, **kwargs):
        if config is None:
            window = WindowConfig(width=width or 1280, height=height or 720)
            renderer = RendererConfig(show_grid=show_grid if show_grid is not None else True,
                                      rendermode="lighted")
            config = Config(window=window, renderer=renderer)

        super().__init__(config=config, **kwargs)
        self.artists = []

    def add(self, data, **kwargs):
        """
        Add a data object to the viewer.

        This is a convenience shortcut for ``viewer.scene.add`` that additionally
        routes a :class:`jax_fdm.datastructures.FDNetwork` through an
        :class:`FDNetworkViewerArtist`. ``compas_viewer.Viewer`` itself has no
        ``add`` method (objects go through ``viewer.scene``), so this wrapper
        provides the terser ``viewer.add(obj)`` interface.

        Parameters
        ----------
        data : :class:`compas.geometry.Geometry` | :class:`compas.datastructures.Datastructure` | :class:`jax_fdm.datastructures.FDNetwork`
            The object to visualize.
        as_wireframe : bool, optional
            If ``True`` and ``data`` is an ``FDNetwork``, draw it as a plain
            wireframe graph instead of the full force-density artist.
        **kwargs : dict, optional
            Additional visualization options.

        Returns
        -------
        The created scene object, or the :class:`FDNetworkViewerArtist` for an
        ``FDNetwork``.
        """
        as_wireframe = kwargs.pop("as_wireframe", False)

        if not isinstance(data, FDNetwork) or as_wireframe:
            return self.scene.add(data, **kwargs)

        artist = FDNetworkViewerArtist(data, viewer=self, **kwargs)
        self.artists.append(artist)
        artist.draw()
        artist.add()

        return artist
