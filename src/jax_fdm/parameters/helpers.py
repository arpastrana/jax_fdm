from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable

import jax
import jax.numpy as jnp
import numpy as np


def split_parameters(
    parray: jax.Array,
    func: Callable[[jax.Array], tuple[np.ndarray, ...]],
) -> tuple[list[jax.Array], np.ndarray]:
    """
    Split a flat array into flat subarrays given a filter function.
    """
    masks = func(parray)
    indices = [np.flatnonzero(mask) for mask in masks]
    sarrays = [parray[idx] for idx in indices]
    adef = np.argsort(np.concatenate(indices))

    return sarrays, adef


def combine_parameters(parrays: tuple[jax.Array, ...], adef: np.ndarray) -> jax.Array:
    """
    Combine a sequence of flat arrays given a filter function.
    """
    return jnp.concatenate(parrays)[adef]


def reshape_parameters(
    sarrays: Iterable[jax.Array],
    shapes: Iterable[tuple[int, ...]],
) -> Generator[jax.Array, None, None]:
    """
    Reshape a sequence of flat arrays.
    """
    return (jnp.reshape(sarray, shape) for sarray, shape in zip(sarrays, shapes))
