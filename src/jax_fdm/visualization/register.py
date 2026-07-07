from jax_fdm.visualization.backends import has_backend

__all__ = ["register_fd_scene_objects"]


def register_fd_scene_objects():
    """
    Register the force density scene objects with compas.scene, per installed backend.

    This makes a native COMPAS scene (a bare :class:`compas_viewer.Viewer` or a
    compas_notebook scene) render an :class:`jax_fdm.datastructures.FDNetwork`
    or :class:`jax_fdm.datastructures.FDMesh` through the jax_fdm scene objects.

    The built-in plugin discovery must run first: compas only auto-discovers
    scene objects when its registry is empty, so registering the force density
    types into a fresh registry would permanently mask every built-in type.
    jax_fdm also cannot register through the plugin system itself, because
    discovery only scans packages whose name starts with "compas".
    """
    if not (has_backend("compas_notebook") or has_backend("compas_viewer")):
        return

    from compas.scene.context import register_scene_objects
    register_scene_objects()

    if has_backend("compas_notebook"):
        from jax_fdm.visualization.notebooks.scene_objects import register_notebook_scene_objects
        register_notebook_scene_objects()

    if has_backend("compas_viewer"):
        from jax_fdm.visualization.viewers.scene_objects import register_viewer_scene_objects
        register_viewer_scene_objects()
