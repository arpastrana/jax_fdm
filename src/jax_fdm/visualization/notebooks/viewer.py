from compas_notebook.config import CameraConfig
from compas_notebook.config import Config
from compas_notebook.config import ViewConfig
from compas_notebook.viewer import Viewer as CompasNotebookViewer

from compas.datastructures import Graph
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization.notebooks.scene import ThreeFDMeshObject
from jax_fdm.visualization.notebooks.scene import ThreeFDNetworkObject

__all__ = ["NotebookViewer"]


class NotebookViewer(CompasNotebookViewer):
    """
    A thin wrapper on :class:`compas_notebook.viewer.Viewer`.

    It subclasses the COMPAS notebook viewer so that the scene, the toolbar and
    ``show`` all work natively, and only overrides ``add`` to route the force
    density datastructures through their notebook artists.

    For convenience it also accepts the ``width``, ``height``, ``show_grid``,
    ``viewport``, ``camera_position`` and ``camera_target`` keyword arguments
    directly and folds them into a :class:`compas_notebook.config.Config`, so
    the common setup does not require building a config by hand.
    """
    def __init__(self,
                 width=None,
                 height=None,
                 show_grid=None,
                 viewport=None,
                 camera_position=None,
                 camera_target=None,
                 config=None,
                 **kwargs):
        if config is None:
            config = Config()
            # Config.view is a shared class attribute in compas_notebook 0.11;
            # assign a fresh ViewConfig so settings do not leak across viewers.
            config.view = ViewConfig(viewport=viewport or "perspective",
                                     width=width or 1100,
                                     height=height or 580,
                                     show_grid=show_grid if show_grid is not None else True)
            if camera_position or camera_target:
                # ViewConfig.__post_init__ resets the camera from the viewport,
                # so the camera overrides land after construction.
                config.view.camera = CameraConfig(position=list(camera_position) if camera_position else config.view.camera.position,
                                                  target=list(camera_target) if camera_target else config.view.camera.target)

        super().__init__(config=config, **kwargs)
        self.artists = []

    def add(self, data, **kwargs):
        """
        Add a data object to the viewer.

        This is a convenience shortcut for ``viewer.scene.add`` that additionally
        routes a :class:`jax_fdm.datastructures.FDNetwork` through an
        :class:`FDNetworkNotebookArtist` and a :class:`jax_fdm.datastructures.FDMesh`
        through an :class:`FDMeshNotebookArtist`.

        Dispatch is purely by type. To draw a force density datastructure as plain
        geometry instead, convert it to its COMPAS base first and add that, e.g.
        ``viewer.add(mesh.copy(cls=Mesh))`` or ``viewer.add(network.copy(cls=Network))``.

        Parameters
        ----------
        data : :class:`compas.geometry.Geometry` | :class:`compas.datastructures.Datastructure` | :class:`jax_fdm.datastructures.FDNetwork` | :class:`jax_fdm.datastructures.FDMesh`
            The object to visualize.
        **kwargs : dict, optional
            Additional visualization options.

        Returns
        -------
        The created scene object, or the notebook artist for an ``FDNetwork`` /
        ``FDMesh``.
        """
        if isinstance(data, FDMesh):
            return self._add_sceneobject(ThreeFDMeshObject, data, **kwargs)

        if isinstance(data, FDNetwork):
            return self._add_sceneobject(ThreeFDNetworkObject, data, **kwargs)

        # COMPAS 2.x aliases ``Network`` to ``Graph``; default the display name
        # to "Network" to mirror the 3D viewer wrapper.
        if isinstance(data, Graph) and "name" not in kwargs:
            kwargs["name"] = "Network"

        return self.scene.add(data, **kwargs)

    def _add_sceneobject(self, cls, data, **kwargs):
        """
        Add a force density datastructure to the scene through its adapter.
        """
        obj = self.scene.add(data, sceneobject_type=cls, **kwargs)
        self.artists.append(obj.artist)

        return obj.artist
