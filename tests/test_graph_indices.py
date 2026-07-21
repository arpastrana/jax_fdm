"""
Tests for `indices_from_keys`, the loop-free key resolver behind goals and
constraints.

The resolver replaced a per-key dict lookup with a vectorized `argsort` +
`searchsorted`. These tests pin that it matches the dict on valid keys, keeps
edge pairs order-sensitive (the directed-edge convention), and raises on any
absent key. The edge case that motivated the last group: edge pairs are folded
to an integer code `u * base + v`, and the fold must stay injective even for an
absent edge whose endpoint runs past the largest canonical node, which a base
sized to the canonical edges alone would let alias onto a real edge.
"""

import numpy as np
import pytest

from jax_fdm.equilibrium import indices_from_keys


def _dict_nodes(canonical, query):
    """
    Resolve node keys with the reference dict lookup the vectorization replaced.
    """
    lookup = {int(k): i for i, k in enumerate(canonical)}

    return [lookup[int(k)] for k in np.asarray(query).ravel()]


def _dict_edges(canonical, query):
    """
    Resolve edge keys with the reference dict lookup the vectorization replaced.
    """
    lookup = {(int(u), int(v)): i for i, (u, v) in enumerate(canonical)}
    pairs = np.asarray(query).reshape(-1, 2)

    return [lookup[(int(u), int(v))] for u, v in pairs]


# ==============================================================================
# Nodes
# ==============================================================================


def test_nodes_match_dict_on_shuffled_noncontiguous_keys():
    """
    The resolver matches the dict lookup on shuffled, non-contiguous node keys.
    """
    nodes = np.array([10, 3, 7, 99, 42, 5])

    for query in [10, 5, 99, [42, 3, 7], [5, 5, 10]]:
        resolved = np.atleast_1d(np.asarray(indices_from_keys(nodes, query)))
        assert resolved.tolist() == _dict_nodes(nodes, query)


def test_nodes_preserve_query_order():
    """
    The resolved indices come back in the order the keys were queried.
    """
    nodes = np.array([10, 3, 7, 99, 42, 5])
    query = [99, 3, 42]

    resolved = np.asarray(indices_from_keys(nodes, query)).tolist()

    assert resolved == _dict_nodes(nodes, query)


@pytest.mark.parametrize("absent", [123, 1000, 1])
def test_absent_node_raises(absent):
    """
    A node key absent from the ordering raises, whether above, below, or between.
    """
    nodes = np.array([10, 3, 7, 99, 42, 5])

    with pytest.raises(KeyError):
        indices_from_keys(nodes, absent)


# ==============================================================================
# Edges
# ==============================================================================


def test_edges_match_dict_and_are_order_sensitive():
    """
    Edge pairs resolve like the dict, and the fold respects edge direction.
    """
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0], [10, 2]])

    for query in [(1, 2), [(0, 1), (10, 2), (3, 0)], (2, 3)]:
        resolved = np.atleast_1d(np.asarray(indices_from_keys(edges, query)))
        assert resolved.tolist() == _dict_edges(edges, query)


def test_reversed_edge_raises():
    """
    A reversed edge is a different key: (2, 1) is absent when only (1, 2) exists.
    """
    edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0]])

    with pytest.raises(KeyError):
        indices_from_keys(edges, (2, 1))


def test_absent_edge_past_canonical_max_raises():
    """
    An absent edge whose endpoint exceeds the largest canonical node still raises.

    Regression: the edge fold `u * base + v` sized `base` from the canonical
    edges alone. With canonical edges (0, 1) and (2, 0), base was 3, so the
    absent edge (1, 3) folded to 1 * 3 + 3 = 6, the same code as (2, 0), and
    resolved to index 1 as a false match instead of raising.
    """
    edges = np.array([[0, 1], [2, 0]])

    with pytest.raises(KeyError):
        indices_from_keys(edges, (1, 3))


def test_edges_fuzz_matches_dict_and_raises_on_absent():
    """
    Across random graphs, the resolver matches the dict on present edges and
    raises on absent ones, including queries with out-of-range endpoints.
    """
    rng = np.random.default_rng(0)

    for _ in range(500):
        num_nodes = int(rng.integers(2, 12))
        num_edges = int(rng.integers(1, 20))
        raw = np.stack(
            [
                rng.integers(0, num_nodes, num_edges),
                rng.integers(0, num_nodes, num_edges),
            ],
            axis=1,
        )

        seen = {}
        for u, v in raw:
            seen.setdefault((int(u), int(v)), len(seen))
        canonical = np.array(list(seen.keys()))

        for key, index in list(seen.items())[:3]:
            resolved = int(np.asarray(indices_from_keys(canonical, key))[0])
            assert resolved == index

        for _ in range(5):
            query = (
                int(rng.integers(0, num_nodes + 5)),
                int(rng.integers(0, num_nodes + 5)),
            )
            if query in seen:
                resolved = int(np.asarray(indices_from_keys(canonical, query))[0])
                assert resolved == seen[query]
            else:
                with pytest.raises(KeyError):
                    indices_from_keys(canonical, query)
