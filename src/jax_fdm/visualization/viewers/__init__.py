from jax_fdm.visualization.backends import has_backend
from jax_fdm.visualization.backends import null_viewer

# The 3D viewer builds on compas_viewer, an optional dependency.
if has_backend("compas_viewer"):
    from compas.scene.context import register_scene_objects

    from .scene_objects import *  # noqa F403
    from .scene_objects import register_viewer_scene_objects
    from .viewer import *  # noqa F403

    # Built-in plugin discovery must run first: compas only auto-discovers into
    # an empty registry, and jax_fdm cannot register via plugins (discovery
    # scans only compas* packages).
    register_scene_objects()
    register_viewer_scene_objects()
else:
    Viewer = null_viewer("compas_viewer")

__all__ = [name for name in dir() if not name.startswith('_')]
