import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array
from jaxtyping import Int

__all__ = [
    "indices_from_keys",
]


def _indices_from_keys(
    keys_canonical: Int[np.ndarray, "elements ..."],
    keys_query: Int[np.ndarray, "..."]
    | int
    | tuple[int, int]
    | list[int]
    | list[tuple[int, int]],
) -> Int[np.ndarray, "queries"]:
    """
    Resolve element keys to their positions in a canonical ordering, loop-free.

    Parameters
    ----------
    keys_canonical :
        The structure's canonical key ordering: a 1-D array of node/vertex/face
        keys, or a 2-D array of edge key pairs, one row per element.
    keys_query :
        The key or keys to resolve, matching the canonical kind. A single node
        key, an edge key pair, or a sequence (a list or a tuple) of either.

    Returns
    -------
    indices :
        The position of each queried key in the canonical ordering.

    Raises
    ------
    KeyError
        If any queried key is absent from the canonical ordering.

    Notes
    -----
    A legacy pure-NumPy resolver that runs off-trace on concrete keys, kept for
    the constraints, which still resolve their keys at build time. Goals use the
    traceable `indices_from_keys` instead.

    Vectorized with ``argsort`` + ``searchsorted`` rather than a per-key dict
    lookup. Edge pairs are folded to a single integer code ``u * base + v`` so
    the same one-dimensional search resolves them; the code is order-sensitive,
    matching the directed edge convention of the connectivity.
    """
    canonical = np.asarray(keys_canonical)
    query = np.asarray(keys_query)

    if canonical.ndim == 2:
        # edge keys: fold each (u, v) pair to an order-sensitive integer code.
        # The base must exceed every node index in both the canonical edges and
        # the queried ones; sized to the canonical edges alone, an absent edge
        # whose endpoint runs past the canonical maximum could fold onto a real
        # edge's code and resolve as a false match instead of raising.
        query = query.reshape(-1, canonical.shape[1])
        base = int(canonical.max())
        if query.size:
            base = max(base, int(query.max()))
        base += 1
        canonical = canonical[:, 0].astype(np.int64) * base + canonical[:, 1]
        query = query[:, 0].astype(np.int64) * base + query[:, 1]
    else:
        query = query.ravel()

    order = np.argsort(canonical)
    canonical_sorted = canonical[order]
    position = np.searchsorted(canonical_sorted, query)

    # searchsorted returns an insertion point; clip it into range and confirm the
    # landed key actually equals the query, so a missing key raises rather than
    # aliasing onto a neighbor.
    position_clipped = np.clip(position, 0, canonical_sorted.size - 1)
    if not np.array_equal(canonical_sorted[position_clipped], query):
        raise KeyError("One or more keys are absent from the structure ordering.")

    return order[position]


def indices_from_keys(
    keys_canonical: Int[np.ndarray, "elements ..."],
    keys_query: Int[Array, "..."],
) -> Int[Array, "..."]:
    """
    Resolve traced element keys to positions in a canonical ordering, under a trace.

    Parameters
    ----------
    keys_canonical :
        The structure's canonical key ordering: a 1-D array of node/vertex/face
        keys, or a 2-D array of edge key pairs, one row per element.
    keys_query :
        The key or keys to resolve, a traced JAX array matching the canonical
        kind: one element's key, mapped one element at a time under a ``vmap``,
        so a node key is a scalar, an edge key a ``(2,)`` pair, and an aggregate's
        key the whole ``(points,)`` or ``(points, 2)`` row.

    Returns
    -------
    indices :
        The position of each queried key in the canonical ordering, its leading
        query shape preserved.

    Notes
    -----
    The traceable twin of `_indices_from_keys`, for resolving a goal's key inside
    the evaluation ``vmap`` where the key is a tracer that numpy cannot consume.
    The canonical side is static topology, so its sort and edge fold run in numpy
    at trace time; only the query search and gather run in JAX. Edge pairs are
    folded to an order-sensitive code ``u * base + v`` with ``base`` sized from
    the canonical edges alone, since the query cannot be measured under a trace;
    the missing-key guard then compares the landed edge's full pair against the
    query, so a query endpoint past the canonical maximum is caught by the pair
    comparison rather than a fold collision.

    A missing key is caught by `equinox.error_if`, which raises at evaluation
    time rather than at trace time, since the traced query has no concrete value
    to check while the trace is built.
    """
    canonical = np.asarray(keys_canonical)

    if canonical.ndim == 2:
        base = int(canonical.max()) + 1
        canonical_code = canonical[:, 0].astype(np.int64) * base + canonical[:, 1]
        query_code = keys_query[..., 0] * base + keys_query[..., 1]
    else:
        canonical_code = canonical.astype(np.int64)
        query_code = keys_query

    order = np.argsort(canonical_code)
    canonical_sorted = canonical_code[order]

    order_jax = jnp.asarray(order)
    canonical_sorted_jax = jnp.asarray(canonical_sorted)

    position = jnp.searchsorted(canonical_sorted_jax, query_code)
    position = jnp.clip(position, 0, canonical_sorted_jax.size - 1)
    index = order_jax[position]

    # Confirm the landed key equals the query, comparing the full edge pair (not
    # the fold code) so a query endpoint past the canonical maximum cannot alias.
    landed = jnp.asarray(canonical)[index]
    missing = jnp.any(landed != keys_query)
    index = eqx.error_if(
        index,
        missing,
        "One or more goal keys are absent from the structure ordering.",
    )

    return index
