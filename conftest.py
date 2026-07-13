"""
Root pytest configuration.

``--doctest-modules`` imports every module under ``src/jax_fdm`` directly,
bypassing the package ``__init__`` files that normally guard the optional
visualization backends. So the backend-dependent modules are skipped from
doctest collection when their backend is not installed, matching the runtime
import guards. The viewer and notebook subpackages are ignored wholesale;
in the plotters subpackage ``loss_plotter.py`` only needs matplotlib, so its
compas_plotter-dependent siblings are ignored by name.
"""

from jax_fdm.visualization.backends import has_backend

collect_ignore = []

if not has_backend("compas_viewer"):
    collect_ignore.append("src/jax_fdm/visualization/viewers")

if not has_backend("compas_notebook"):
    collect_ignore.append("src/jax_fdm/visualization/notebooks")

if not has_backend("compas_plotter"):
    collect_ignore.append("src/jax_fdm/visualization/plotters/scene_objects.py")
    collect_ignore.append("src/jax_fdm/visualization/plotters/plotter.py")
