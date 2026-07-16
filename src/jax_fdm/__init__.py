"""
********************************************************************************
jax_fdm
********************************************************************************

.. currentmodule:: jax_fdm


.. toctree::
    :maxdepth: 1


"""

from __future__ import print_function

import os
from importlib.util import find_spec

import jax.numpy as jnp
import numpy as np

__author__ = ["Rafael Pastrana"]
__copyright__ = "Rafael Pastrana"
__license__ = "MIT License"
__email__ = "arpastrana@princeton.edu"
__version__ = "0.13.0"


HERE = os.path.dirname(__file__)

HOME = os.path.abspath(os.path.join(HERE, "../../"))
DATA = os.path.abspath(os.path.join(HOME, "data"))
DOCS = os.path.abspath(os.path.join(HOME, "docs"))
TEMP = os.path.abspath(os.path.join(HOME, "temp"))

__all__ = ["HOME", "DATA", "DOCS", "TEMP", "has_backend"]


def has_backend(name: str) -> bool:
    """
    Check whether an optional backend package is installed.

    Several backends are optional dependencies: the 3D viewer (``compas_viewer``),
    the notebook viewer (``compas_notebook``), the 2D plotter (``compas_plotter``)
    and the interior-point optimizer (``cyipopt``). Their absence should degrade
    gracefully instead of breaking ``import jax_fdm``.

    Parameters
    ----------
    name : str
        The import name of the backend package.

    Returns
    -------
    bool
        ``True`` if the package can be imported, ``False`` otherwise.
    """
    return find_spec(name) is not None


# config.py
# define floating point precision
DTYPE_NP = np.float64
DTYPE_JAX = jnp.float64

DTYPE_INT_NP = jnp.int64
DTYPE_INT_JAX = jnp.int64

# this only works on startup!
if DTYPE_JAX == jnp.float64 or DTYPE_NP == np.float64:
    import jax

    jax.config.update("jax_enable_x64", True)
