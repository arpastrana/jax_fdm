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

# The notebook backend still targets the compas 1.x notebook API and is pending
# the compas 2.x port, so it is always skipped from doctest collection (its
# module-level imports break on compas_notebook 2.x regardless of installation).
collect_ignore.append("src/jax_fdm/visualization/notebooks/viewer.py")
collect_ignore.append("src/jax_fdm/visualization/notebooks/network_artist.py")
collect_ignore.append("src/jax_fdm/visualization/notebooks/register.py")

if not has_backend("compas_plotters"):
    collect_ignore.append("src/jax_fdm/visualization/plotters/network_artist.py")
    collect_ignore.append("src/jax_fdm/visualization/plotters/vector_artist.py")
    collect_ignore.append("src/jax_fdm/visualization/plotters/plotter.py")
    collect_ignore.append("src/jax_fdm/visualization/plotters/register.py")
