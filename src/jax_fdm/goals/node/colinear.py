import jax.numpy as jnp

from jax_fdm.geometry import colinearity_points
from jax_fdm.geometry import curvature_points

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.node import NodeGoal


class NodesColinearGoal(ScalarGoal, NodeGoal):
    """
    Minimize length-normalized colinearity energy for an ordered sequence of points.
    This goal favors solutions where points are evenly spaced.

    Notes
    -----
    - The goal applies to a *collection* of ordered nodes and is therefore not collectible.
    - The start and end points are assumed to be fixed.
    """
    def __init__(self, key, weight=1.0):
        super().__init__(key=key, target=0.0, weight=weight)
        self.is_collectible = False

    def init(self, model, structure):
        """
        Initialize the goal with information from an equilibrium model.
        """
        self.index = [super().index_from_model(model, structure)]

    def prediction(self, eq_state, index):
        """
        Length-normalized colinearity energy of the ordered points.
        """
        P = eq_state.xyz[index, :]

        return jnp.atleast_1d(colinearity_points(P))


class NodesCurvatureGoal(ScalarGoal, NodeGoal):
    """
    Minimize curvature energy (i.e., the turning angle) for an ordered sequence of points.

    Notes
    -----
    - The goal applies to a *collection* of ordered nodes and is therefore not collectible.
    - The start and end points are assumed to be fixed.
    """
    def __init__(self, key, weight=1.0):
        super().__init__(key=key, target=0.0, weight=weight)
        self.is_collectible = False

    def init(self, model, structure):
        """
        Initialize the goal with information from an equilibrium model.
        """
        self.index = [super().index_from_model(model, structure)]

    def prediction(self, eq_state, index):
        """
        Curvature energy of the ordered points.
        """
        P = eq_state.xyz[index, :]

        return jnp.atleast_1d(curvature_points(P))
