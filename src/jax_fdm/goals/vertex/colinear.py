from jax_fdm.goals.node import NodesColinearGoal
from jax_fdm.goals.node import NodesCurvatureGoal
from jax_fdm.goals.vertex import VertexGoal


class VerticesColinearGoal(VertexGoal, NodesColinearGoal):
    """
    Minimize length-normalized colinearity energy for an ordered sequence of
    vertices. This goal favors solutions where vertices are evenly spaced.

    Notes
    -----
    This goal applies to a *collection* of ordered vertices and is therefore
    not collectible.
    """


class VerticesCurvatureGoal(VertexGoal, NodesCurvatureGoal):
    """
    Minimize curvature energy (i.e., the turning angle) for an ordered sequence
    of vertices.

    Notes
    -----
    This goal applies to a *collection* of ordered vertices and is therefore
    not collectible.
    """
