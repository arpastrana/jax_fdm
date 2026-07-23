from collections.abc import Sequence

from jax_fdm.goals.node.colinear import NodesColinearGoal
from jax_fdm.goals.node.colinear import NodesCurvatureGoal
from jax_fdm.goals.vertex.vertex import VertexGoal

__all__ = [
    "VerticesColinearGoal",
    "VerticesCurvatureGoal",
]


class VerticesColinearGoal(VertexGoal, NodesColinearGoal):
    """
    Minimize length-normalized colinearity energy for an ordered sequence of
    vertices. This goal favors solutions where vertices are evenly spaced.

    Notes
    -----
    This goal applies to a *collection* of ordered vertices and is therefore
    not collectible.
    """

    def __init__(self, key: Sequence[int], weight: float = 1.0) -> None:
        # equinox re-synthesizes an __init__ from the fields for every Module
        # subclass and its synthesized inits do not chain, so the aggregate
        # parent's initializer is invoked explicitly rather than through super().
        NodesColinearGoal.__init__(self, key=key, weight=weight)


class VerticesCurvatureGoal(VertexGoal, NodesCurvatureGoal):
    """
    Minimize curvature energy (i.e., the turning angle) for an ordered sequence
    of vertices.

    Notes
    -----
    This goal applies to a *collection* of ordered vertices and is therefore
    not collectible.
    """

    def __init__(self, key: Sequence[int], weight: float = 1.0) -> None:
        # equinox re-synthesizes an __init__ from the fields for every Module
        # subclass and its synthesized inits do not chain, so the aggregate
        # parent's initializer is invoked explicitly rather than through super().
        NodesCurvatureGoal.__init__(self, key=key, weight=weight)
