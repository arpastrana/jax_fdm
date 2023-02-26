import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeForceGoal(ScalarGoal, EdgeGoal):
    """
    Make an edge of a network to reach a target force.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The predicted edge force.
        """
        return eq_state.forces[index]


class EdgesForceEqualGoal(ScalarGoal, EdgeGoal):
    """
    Equalize the internal force in a selection of edges by minimizing
    the normalized variance of their internal forces.
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
        The normalized variance of the forces of the edges.
        """
        forces = eq_state.forces[index]

        return jnp.atleast_1d(jnp.var(forces) / jnp.mean(forces))
