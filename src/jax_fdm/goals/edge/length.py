import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeLengthGoal(ScalarGoal, EdgeGoal):
    """
    Make an edge of a network to reach a certain length.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The current edge length.
        """
        return eq_state.lengths[index]


class EdgesLengthEqualGoal(ScalarGoal, EdgeGoal):
    """
    Equalize the length of a selection of edges by minimizing
    the normalized variance of their lengths.
    """
    def __init__(self, key, weight=1.0):
        super().__init__(key=key, target=0.0, weight=weight)
        self.is_collectible = False

    def init(self, model):
        """
        Initialize the goal with information from an equilibrium model.
        """
        self.index = [super().index_from_model(model)]

    @staticmethod
    def prediction(eq_state, index):
        """
        The normalized variance of the lengths of the edges.
        """
        lengths = eq_state.lengths[index]

        return jnp.atleast_1d(jnp.var(lengths) / jnp.mean(lengths))
