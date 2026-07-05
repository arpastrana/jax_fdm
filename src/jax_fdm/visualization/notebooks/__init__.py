from jax_fdm.visualization.backends import has_backend
from jax_fdm.visualization.backends import null_viewer

# The notebook viewer builds on compas_notebook, an optional dependency.
#
# NOTE: this backend still targets the compas 1.x notebook API and is pending
# the compas 2.x port. compas_notebook 2.x removed ``compas_notebook.app``, so
# the legacy import below raises on a 2.x install; degrade to the null viewer
# in that case so ``import jax_fdm.visualization`` keeps working until the port
# lands.
if has_backend("compas_notebook"):
    try:
        from .shapes import *  # noqa F403
        from .network_artist import *  # noqa F403
        from .viewer import *  # noqa F403
        from .register import register_artists

        register_artists()
    except ImportError:
        NotebookViewer = null_viewer("compas_notebook")
else:
    NotebookViewer = null_viewer("compas_notebook")

__all__ = [name for name in dir() if not name.startswith('_')]
