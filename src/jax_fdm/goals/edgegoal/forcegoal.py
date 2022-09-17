import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edgegoal import EdgeGoal


class EdgeForceGoal(ScalarGoal, EdgeGoal):
    """
    Make an edge of a network to reach a certain force.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def prediction(self, eq_state):
        """
        The current edge length.
        """
        force = eq_state.forces[self.index, ]
        return jnp.atleast_1d(force)
