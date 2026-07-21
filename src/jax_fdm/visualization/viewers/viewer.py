from typing import Any
from typing import Callable

import compas_viewer.components.sidebar
from compas_viewer import Viewer as CompasViewer
from compas_viewer.config import Config
from compas_viewer.config import RendererConfig
from compas_viewer.config import WindowConfig

from compas.datastructures import Datastructure
from compas.datastructures import Graph
from compas.geometry import Geometry
from compas.scene import SceneObject
from jax_fdm.datastructures import FDMesh
from jax_fdm.datastructures import FDNetwork
from jax_fdm.visualization.viewers.buffer_manager import FastBufferManager
from jax_fdm.visualization.viewers.scene_objects import FDObject
from jax_fdm.visualization.viewers.sidebar import FDObjectSetting

__all__ = ["Viewer"]


class Viewer(CompasViewer):
    """
    A thin wrapper on `compas_viewer.Viewer`.

    It subclasses the COMPAS viewer so that camera control, the ``on`` frame
    loop, ``show`` and recording all work natively. The force density
    datastructures render through their registered scene objects, so ``add``
    only provides the terser ``viewer.add(obj)`` interface and a couple of
    kwarg conveniences.

    For convenience it also accepts the ``width``, ``height`` and ``show_grid``
    keyword arguments directly and folds them into a
    `compas_viewer.config.Config`,
    so the common window setup does not require building a config by hand.
    The defaults (1200x800, no grid) fit a typical laptop screen and keep the
    grid from cutting through structures that hang below ``z=0``.

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

    def __init__(
        self,
        width: int | None = None,
        height: int | None = None,
        show_grid: bool | None = None,
        config: Config | None = None,
        **kwargs: Any,
    ) -> None:
        if config is None:
            window = WindowConfig(width=width or 1200, height=height or 800)
            renderer = RendererConfig(
                show_grid=show_grid if show_grid is not None else False,
                rendermode="lighted",
            )
            config = Config(window=window, renderer=renderer)

        # The sidebar constructs its object settings tab from the name in its
        # module globals, so swapping the class in for the duration of the
        # construction slots the force density readout into the native tab.
        sidebar = compas_viewer.components.sidebar
        native_setting = sidebar.ObjectSetting
        sidebar.ObjectSetting = FDObjectSetting
        try:
            super().__init__(config=config, **kwargs)
        finally:
            sidebar.ObjectSetting = native_setting

        # Swap in the vectorized buffer manager before any GL buffers exist.
        self.renderer.buffer_manager = FastBufferManager()
        self._warned_unfused_on = False

        # Clamp the window to the screen: the upstream centering math places
        # an oversized window at a negative origin, which Qt warns about and
        # snaps back to the primary screen.
        rect = self.app.primaryScreen().availableGeometry()
        window = self.config.window
        if window.width > rect.width() or window.height > rect.height():
            width, height = window.width, window.height
            window.width = min(width, rect.width())
            window.height = min(height, rect.height())
            print(
                f"WARNING: The window size {width}x{height} exceeds the available "
                f"screen space. Resizing the window to {window.width}x{window.height}",
            )

    def clear(self) -> None:
        """
        Empty the scene for the next round of adds.

        Call between sequential shows, while the window is closed. Besides the
        scene objects this also drops the picking-color registrations, which
        the parent scene never prunes.
        """
        self.scene.clear()
        self.scene.instance_colors.clear()

    def update(self) -> None:
        """
        Repaint the viewer window immediately.

        Schedules a repaint and drains the event queue so it happens now,
        painting whatever the scene objects currently hold. This acts at the
        window layer: it does not touch scene data, so refresh the buffers of
        any mutated object first (e.g. ``scene_object.update()``).

        Notes
        -----
        The intended use is animating a blocking computation that runs on the
        GUI thread. Because the computation blocks Qt's event loop, a normally
        scheduled repaint would not be serviced until it returns and every frame
        would collapse into one. Calling this per step forces each frame to
        paint while the computation is still running. A computation on a worker
        thread does not need it: the viewer repaints on its own timer.
        """
        self.renderer.update()
        self.app.processEvents()

    def show(self) -> None:
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

    def on(
        self,
        interval: int,
        frames: int | None = None,
    ) -> Callable[[Callable[..., None]], Callable[..., None]]:
        """
        Decorate a frame callback for the animation loop.

        Warns once when per-element scene objects are in the scene: their
        buffers update one by one with a full-buffer index lookup each, per
        frame, so animation slows quadratically with element count. Re-add
        the datastructure with ``fuse=True`` to animate on batched soups.
        """
        if not self._warned_unfused_on:
            if any(isinstance(obj, FDObject) for obj in self.scene.objects):
                print(
                    "Animating per-element scene objects updates every element buffer "
                    "one by one, each frame; re-add with viewer.add(..., fuse=True) "
                    "to animate on batched soups instead.",
                )
                self._warned_unfused_on = True

        return super().on(interval, frames)

    def add(self, data: Geometry | Datastructure, **kwargs: Any) -> SceneObject:
        """
        Add a data object to the viewer.

        This is a convenience shortcut for ``viewer.scene.add`` with a couple
        of kwarg conveniences for plain COMPAS objects. The force density
        datastructures dispatch through the scene registry to their scene
        objects (`FDNetworkObject`, `FDMeshObject`).

        Dispatch is purely by type. To draw a force-density datastructure as plain
        geometry instead (a bare wireframe or shaded surface), convert it to its
        COMPAS base first and add that, e.g. ``viewer.add(mesh.copy(cls=Mesh))``
        or ``viewer.add(network.copy(cls=Network))``.

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
        if not isinstance(data, (FDMesh, FDNetwork)):
            # COMPAS 2.x aliases ``Network`` to ``Graph``, so a plain network added
            # as a reference wireframe would otherwise show up as "Graph" in the
            # scene tree. Default its display name to "Network".
            if isinstance(data, Graph) and "name" not in kwargs:
                kwargs["name"] = "Network"

            # compas_viewer's line width kwarg is ``linewidth`` (screen-space
            # pixels); the ``edgewidth`` it inherits from compas.scene is stored
            # but never consumed by the render pipeline. Alias it so ``edgewidth``
            # is the one edge-width vocabulary across FD and plain adds alike.
            if "edgewidth" in kwargs and "linewidth" not in kwargs:
                kwargs["linewidth"] = kwargs.pop("edgewidth")

        return self.scene.add(data, **kwargs)
