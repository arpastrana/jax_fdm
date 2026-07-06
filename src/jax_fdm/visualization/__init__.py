from .artists import *  # noqa F403
from .plotters import *  # noqa F403
from .viewers import *  # noqa F403
from .notebooks import *  # noqa F403
from .scene import register_fd_scene_objects

# Make native COMPAS scenes render the force density datastructures through
# the jax_fdm artists in the installed backend contexts.
register_fd_scene_objects()

__all__ = [name for name in dir() if not name.startswith('_')]
