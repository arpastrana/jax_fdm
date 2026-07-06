"""
Root pytest configuration.

``--doctest-modules`` imports every module under ``src/jax_fdm`` directly,
bypassing the package ``__init__`` files that normally guard the optional
visualization backends. So the viewer modules that import ``compas_viewer`` or
``compas_notebook`` at module level are skipped from doctest collection when
their backend is not installed, matching the runtime import guards.
"""

from jax_fdm.visualization.backends import has_backend

collect_ignore = []

if not has_backend("compas_viewer"):
    collect_ignore.append("src/jax_fdm/visualization/viewers/viewer.py")
    collect_ignore.append("src/jax_fdm/visualization/viewers/network_artist.py")

if not has_backend("compas_notebook"):
    collect_ignore.append("src/jax_fdm/visualization/notebooks/datastructure_artist.py")
    collect_ignore.append("src/jax_fdm/visualization/notebooks/network_artist.py")
    collect_ignore.append("src/jax_fdm/visualization/notebooks/mesh_artist.py")
    collect_ignore.append("src/jax_fdm/visualization/notebooks/scene.py")
    collect_ignore.append("src/jax_fdm/visualization/notebooks/viewer.py")

if not has_backend("compas_plotters"):
    collect_ignore.append("src/jax_fdm/visualization/plotters/network_artist.py")
    collect_ignore.append("src/jax_fdm/visualization/plotters/vector_artist.py")
    collect_ignore.append("src/jax_fdm/visualization/plotters/plotter.py")
    collect_ignore.append("src/jax_fdm/visualization/plotters/register.py")
