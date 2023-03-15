import numpy as np
import jax.numpy as jnp

from jax import jit


def split_parameters(parray, func):
    """
    Split a flat array into flat subarrays given a filter function.
    """
    masks = func(parray)
    indices = [np.flatnonzero(mask) for mask in masks]
    sarrays = [parray[idx] for idx in indices]
    adef = np.argsort(np.concatenate(indices))

    return sarrays, adef


def combine_parameters(parrays, adef):
    """
    Combine a sequence of flat arrays given a filter function.
    """
    return jnp.concatenate(parrays)[adef]


def reshape_parameters(sarrays, shapes):
    """
    Reshape a sequence of flat arrays.
    """
    return (jnp.reshape(sarray, shape) for sarray, shape in zip(sarrays, shapes))
