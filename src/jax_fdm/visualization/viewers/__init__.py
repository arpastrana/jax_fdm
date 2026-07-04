from jax_fdm.visualization.backends import has_backend
from jax_fdm.visualization.backends import null_viewer

if has_backend("compas_view2"):
    from .network_artist import *  # noqa F403
    from .viewer import *  # noqa F403
    from .register import register_artists

    register_artists()
else:
    Viewer = null_viewer("compas_view2")

__all__ = [name for name in dir() if not name.startswith('_')]
