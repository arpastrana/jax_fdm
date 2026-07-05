# The base artists are plain, COMPAS-free classes (they no longer subclass the
# removed ``compas.artists.NetworkArtist``), so they always import regardless of
# which visualization backend, if any, is installed.
from .datastructure_artist import *  # noqa F403
from .network_artist import *  # noqa F403
from .mesh_artist import *  # noqa F403

__all__ = [name for name in dir() if not name.startswith('_')]
