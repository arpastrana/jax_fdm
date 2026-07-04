from jax_fdm.visualization.backends import has_backend

# The base artist subclasses ``compas.artists.NetworkArtist``, which only
# exists on compas 1.x (removed in 2.x in favor of ``compas.scene``). All
# backend artists build on it, and every backend is itself compas-1.x-only,
# so guard the whole base behind the presence of the 1.x artist API.
if has_backend("compas.artists"):
    from .network_artist import *  # noqa F403

__all__ = [name for name in dir() if not name.startswith('_')]
