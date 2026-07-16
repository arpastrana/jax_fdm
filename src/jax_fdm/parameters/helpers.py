from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int
from jaxtyping import Shaped


def split_parameters(
    parray: Float[Array, "parameters"],
    func: Callable[
        [Float[Array, "parameters"]],
        tuple[Shaped[np.ndarray, "parameters"], ...],
    ],
) -> tuple[list[Float[Array, "..."]], Int[np.ndarray, "parameters"]]:
    """
    Split a flat array into flat subarrays given a filter function.
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
    Combine a sequence of flat arrays given a filter function.
    """
    return jnp.concatenate(parrays)[adef]


def reshape_parameters(
    sarrays: Iterable[Float[Array, "..."]],
    shapes: Iterable[tuple[int, ...]],
) -> Iterator[Float[Array, "..."]]:
    """
    Reshape a sequence of flat arrays.
    """
    return (jnp.reshape(sarray, shape) for sarray, shape in zip(sarrays, shapes))
