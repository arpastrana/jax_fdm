from jax_fdm.visualization.backends import has_backend
from jax_fdm.visualization.backends import null_viewer

# The notebook viewer builds on compas_notebook, an optional dependency.
if has_backend("compas_notebook"):
    from .shapes import *  # noqa F403
    from .network_artist import *  # noqa F403
    from .viewer import *  # noqa F403
    from .register import register_artists

    register_artists()
else:
    NotebookViewer = null_viewer("compas_notebook")

__all__ = [name for name in dir() if not name.startswith('_')]
