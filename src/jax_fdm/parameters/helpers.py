from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Shaped

__all__ = [
    "combine_parameters",
    "reshape_parameters",
    "split_parameters",
]


def split_parameters(
    parray: Float[Array, "parameters"],
    func: Callable[
        [Float[Array, "parameters"]],
        tuple[Shaped[np.ndarray, "parameters"], ...],
    ],
) -> tuple[list[Float[Array, "..."]], Int[np.ndarray, "parameters"]]:
    """
    Split a flat array into subarrays selected by a masking function.

    Parameters
    ----------
    parray :
        The flat parameter array to split.
    func :
        A function returning one boolean mask per output subarray.

    Returns
    -------
    split :
        The list of masked subarrays, and the permutation that reverses the split
        so :func:`combine_parameters` can restore the original order.
    """
    masks = func(parray)
    indices = [np.flatnonzero(mask) for mask in masks]
    sarrays = [parray[idx] for idx in indices]
    adef = np.argsort(np.concatenate(indices))

    return sarrays, adef


def combine_parameters(
    parrays: tuple[Float[Array, "..."], ...],
    adef: Int[np.ndarray, "parameters"],
) -> Float[Array, "parameters"]:
    """
    Merge subarrays back into one flat array, inverting a split.

    Parameters
    ----------
    parrays :
        The subarrays to concatenate, in the order they were split.
    adef :
        The permutation returned by :func:`split_parameters` that restores the
        original element order.

    Returns
    -------
    parray :
        The recombined flat parameter array.
    """
    return jnp.concatenate(parrays)[adef]


def reshape_parameters(
    sarrays: Iterable[Float[Array, "..."]],
    shapes: Iterable[tuple[int, ...]],
) -> Iterator[Float[Array, "..."]]:
    """
    Reshape each flat array to its paired target shape.

    Parameters
    ----------
    sarrays :
        The flat arrays to reshape.
    shapes :
        The target shape for each array, paired by position.

    Yields
    ------
    array :
        Each input array reshaped to its target shape.
    """
    return (jnp.reshape(sarray, shape) for sarray, shape in zip(sarrays, shapes))
