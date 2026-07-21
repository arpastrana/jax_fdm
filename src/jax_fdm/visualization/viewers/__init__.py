from jax_fdm.visualization.backends import has_backend
from jax_fdm.visualization.backends import null_viewer

__all__ = []

# The 3D viewer builds on compas_viewer, an optional dependency.
if has_backend("compas_viewer"):
    from compas.scene.context import register_scene_objects

    from .scene_objects import FDDatastructureObject
    from .scene_objects import FDGroupObject
    from .scene_objects import FDMeshObject
    from .scene_objects import FDNetworkObject
    from .scene_objects import FDObject
    from .scene_objects import register_viewer_scene_objects
    from .viewer import Viewer

    # Built-in plugin discovery must run first: compas only auto-discovers into
    # an empty registry, and jax_fdm cannot register via plugins (discovery
    # scans only compas* packages).
    register_scene_objects()
    register_viewer_scene_objects()

    __all__ += [
        "FDDatastructureObject",
        "FDNetworkObject",
        "FDMeshObject",
        "FDGroupObject",
        "FDObject",
        "register_viewer_scene_objects",
        "Viewer",
    ]
else:
    Viewer = null_viewer("compas_viewer")
    __all__ += ["Viewer"]
