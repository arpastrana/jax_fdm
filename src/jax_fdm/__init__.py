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

import numpy as np
import jax.numpy as jnp


__author__ = ["Rafael Pastrana"]
__copyright__ = "Rafael Pastrana"
__license__ = "MIT License"
__email__ = "arpastrana@princeton.edu"
__version__ = "0.5.0"


HERE = os.path.dirname(__file__)

HOME = os.path.abspath(os.path.join(HERE, "../../"))
DATA = os.path.abspath(os.path.join(HOME, "data"))
DOCS = os.path.abspath(os.path.join(HOME, "docs"))
TEMP = os.path.abspath(os.path.join(HOME, "temp"))

__all__ = ["HOME", "DATA", "DOCS", "TEMP"]

# config.py
# define floating point precision
DTYPE_NP = np.float64
DTYPE_JAX = jnp.float64

# this only works on startup!
if DTYPE_JAX == jnp.float64 or DTYPE_NP == np.float64:
    from jax.config import config
    config.update("jax_enable_x64", True)
