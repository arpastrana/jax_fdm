from typing import Any
from typing import Literal

from compas_notebook.config import CameraConfig
from compas_notebook.config import Config
from compas_notebook.config import ViewConfig
from compas_notebook.viewer import Viewer as CompasNotebookViewer

from compas.datastructures import Datastructure
from compas.datastructures import Graph
from compas.geometry import Geometry
from compas.scene import SceneObject
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork

__all__ = ["NotebookViewer"]


class NotebookViewer(CompasNotebookViewer):
    """
    A thin wrapper on `compas_notebook.viewer.Viewer`.

    It subclasses the COMPAS notebook viewer so that the scene, the toolbar and
    ``show`` all work natively. The force density datastructures render through
    their registered scene objects, so ``add`` only provides a couple of kwarg
    conveniences.

    For convenience it also accepts the ``width``, ``height``, ``show_grid``,
    ``viewport``, ``camera_position`` and ``camera_target`` keyword arguments
    directly and folds them into a `compas_notebook.config.Config`, so
    the common setup does not require building a config by hand.
    """
    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        show_grid: bool | None = None,
        viewport: Literal["top", "perspective"] | None = None,
        camera_position: list[float] | None = None,
        camera_target: list[float] | None = None,
        config: Config | None = None,
        **kwargs: Any,
    ) -> None:
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

    def add(self, data: Geometry | Datastructure, **kwargs: Any) -> SceneObject:
        """
        Add a data object to the viewer.

        This is a convenience shortcut for ``viewer.scene.add``. The force
        density datastructures dispatch through the scene registry to their
        scene objects (`ThreeFDNetworkObject`, `ThreeFDMeshObject`).

        Dispatch is purely by type. To draw a force density datastructure as plain
        geometry instead, convert it to its COMPAS base first and add that, e.g.
        ``viewer.add(mesh.copy(cls=Mesh))`` or ``viewer.add(network.copy(cls=Network))``.

        Parameters
        ----------
        data :
            The geometry or datastructure to visualize.
        kwargs :
            Additional visualization options passed to the scene object.

        Returns
        -------
        scene_object :
            The created scene object.
        """
        # COMPAS 2.x aliases ``Network`` to ``Graph``; default the display name
        # to "Network" to mirror the 3D viewer wrapper.
        if isinstance(data, Graph) and not isinstance(data, (FDMesh, FDNetwork)) and "name" not in kwargs:
            kwargs["name"] = "Network"

        return self.scene.add(data, **kwargs)
