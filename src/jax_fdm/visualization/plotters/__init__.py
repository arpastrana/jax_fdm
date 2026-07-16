from jax_fdm.visualization.backends import has_backend
from jax_fdm.visualization.backends import null_viewer

# LossPlotter only needs matplotlib, so it is always available.
from .loss_plotter import *  # noqa F403

# The 2D plotter builds on compas_plotter, an optional dependency.
if has_backend("compas_plotter"):
    from compas.scene.context import register_scene_objects

    from .plotter import *  # noqa F403
    from .scene_objects import *  # noqa F403
    from .scene_objects import register_plotter_scene_objects

    # Built-in plugin discovery must run first: compas only auto-discovers into
    # an empty registry, and jax_fdm cannot register via plugins (discovery
    # scans only compas* packages).
    register_scene_objects()
    register_plotter_scene_objects()
else:
    Plotter = null_viewer("compas_plotter")

__all__ = [name for name in dir() if not name.startswith("_")]
