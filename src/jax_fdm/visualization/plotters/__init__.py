from jax_fdm.visualization.backends import has_backend

# LossPlotter only needs matplotlib, so it is always available.
from .loss_plotter import *  # noqa F403

# The 2D plotter artists build on compas_plotters, an optional dependency.
if has_backend("compas_plotters"):
    from .network_artist import *  # noqa F403
    from .vector_artist import *  # noqa F403
    from .plotter import *  # noqa F403
    from .register import register_artists

    register_artists()

__all__ = [name for name in dir() if not name.startswith('_')]
