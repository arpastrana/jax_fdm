from jax import jit
import jax.numpy as jnp
import numpy as np


def split(parray, func):
    """
    Split a flat array into flat subarrays given a filter function.
    """
    masks = func(parray)
    indices = [np.flatnonzero(mask) for mask in masks]
    sarrays = [parray[idx] for idx in indices]
    adef = np.argsort(np.concatenate(indices))

    return sarrays, adef


@jit
def combine(*parrays, adef):
    """
    Combine a sequence of flat arrays given a filter function.
    """
    return jnp.concatenate(parrays)[adef]


@jit
def reshape(sarrays, shapes):
    """
    Reshape a sequence of flat arrays.
    """
    return (jnp.reshape(sarray, shape) for sarray, shape in zip(sarrays, shapes))
