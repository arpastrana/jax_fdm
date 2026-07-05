# The base artist is a plain, COMPAS-free class (it no longer subclasses the
# removed ``compas.artists.NetworkArtist``), so it always imports regardless of
# which visualization backend, if any, is installed.
from .network_artist import *  # noqa F403

__all__ = [name for name in dir() if not name.startswith('_')]
