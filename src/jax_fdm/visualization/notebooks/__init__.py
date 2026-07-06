from jax_fdm.visualization.backends import has_backend
from jax_fdm.visualization.backends import null_viewer

# The notebook viewer builds on compas_notebook (>= 0.11, compas 2.x),
# an optional dependency.
if has_backend("compas_notebook"):
    from .datastructure_artist import *  # noqa F403
    from .network_artist import *  # noqa F403
    from .mesh_artist import *  # noqa F403
    from .scene import *  # noqa F403
    from .viewer import *  # noqa F403
else:
    NotebookViewer = null_viewer("compas_notebook")

__all__ = [name for name in dir() if not name.startswith('_')]
