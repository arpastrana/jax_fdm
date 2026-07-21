from types import ModuleType as _ModuleType

from jax_fdm.visualization.backends import has_backend
from jax_fdm.visualization.backends import null_viewer

# The notebook viewer builds on compas_notebook (>= 0.11, compas 2.x),
# an optional dependency.
if has_backend("compas_notebook"):
    from compas.scene.context import register_scene_objects

    from .scene_objects import *  # noqa: F403
    from .scene_objects import register_notebook_scene_objects
    from .viewer import *  # noqa: F403

    # Built-in plugin discovery must run first: compas only auto-discovers into
    # an empty registry, and jax_fdm cannot register via plugins (discovery
    # scans only compas* packages).
    register_scene_objects()
    register_notebook_scene_objects()
else:
    NotebookViewer = null_viewer("compas_notebook")

__all__ = [
    name
    for name, value in vars().items()
    if not name.startswith("_") and not isinstance(value, _ModuleType)
]
