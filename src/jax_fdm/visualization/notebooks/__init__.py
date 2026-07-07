from jax_fdm.visualization.backends import has_backend
from jax_fdm.visualization.backends import null_viewer

# The notebook viewer builds on compas_notebook (>= 0.11, compas 2.x),
# an optional dependency.
if has_backend("compas_notebook"):
    from compas.scene.context import register_scene_objects

    from .scene_objects import *  # noqa F403
    from .scene_objects import register_notebook_scene_objects
    from .viewer import *  # noqa F403

    # The built-in plugin discovery must run first: compas only auto-discovers
    # scene objects when its registry is empty, so registering the force
    # density types into a fresh registry would permanently mask every
    # built-in type. jax_fdm also cannot register through the plugin system
    # itself, because discovery only scans packages whose name starts with
    # "compas".
    register_scene_objects()
    register_notebook_scene_objects()
else:
    NotebookViewer = null_viewer("compas_notebook")

__all__ = [name for name in dir() if not name.startswith('_')]
