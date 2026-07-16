"""
Root pytest configuration.

``--doctest-modules`` imports every module under ``src/jax_fdm`` directly,
bypassing the package ``__init__`` files that normally guard the optional
visualization backends. So the backend-dependent modules are skipped from
doctest collection when their backend is not installed, matching the runtime
import guards. The viewer and notebook subpackages are ignored wholesale;
in the plotters subpackage ``loss_plotter.py`` only needs matplotlib, so its
compas_plotter-dependent siblings are ignored by name.

The suite is compilation-bound: nearly every test jits an equilibrium model,
so the runtime is dominated by XLA compilation, not computation. Pointing JAX
at a persistent on-disk cache lets one run reuse the kernels the last run
compiled, roughly halving wall time from the second run onward. The cache dir
defaults to a stable per-user location but honours ``JAX_COMPILATION_CACHE_DIR``
so CI can redirect it into an ``actions/cache`` path. Combined with the default
``-n auto`` in ``pyproject.toml``, this is the local + CI acceleration story.
"""

import os
import tempfile

import jax

from jax_fdm.visualization.backends import has_backend

# Persist compiled XLA across runs. The tests recompile the same jitted
# equilibrium models every time, so caching the kernels is the single biggest
# local speedup; CI reuses it by pointing JAX_COMPILATION_CACHE_DIR at a cached
# path. Lower the size/time floors so the small test kernels are actually stored
# (JAX otherwise skips fast-to-compile entries).
_cache_dir = os.environ.get(
    "JAX_COMPILATION_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "jax_fdm_jit_cache"),
)
jax.config.update("jax_compilation_cache_dir", _cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)

collect_ignore = []

if not has_backend("compas_viewer"):
    collect_ignore.append("src/jax_fdm/visualization/viewers")

if not has_backend("compas_notebook"):
    collect_ignore.append("src/jax_fdm/visualization/notebooks")

if not has_backend("compas_plotter"):
    collect_ignore.append("src/jax_fdm/visualization/plotters/scene_objects.py")
    collect_ignore.append("src/jax_fdm/visualization/plotters/plotter.py")
