from jax_fdm.visualization.backends import has_backend
from jax_fdm.visualization.backends import null_viewer

# The 3D viewer builds on compas_viewer, an optional dependency.
if has_backend("compas_viewer"):
    from .network_artist import *  # noqa F403
    from .viewer import *  # noqa F403
else:
    Viewer = null_viewer("compas_viewer")

__all__ = [name for name in dir() if not name.startswith('_')]
