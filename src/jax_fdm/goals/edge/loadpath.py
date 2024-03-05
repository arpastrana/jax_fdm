import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeLoadPathGoal(ScalarGoal, EdgeGoal):
    """
    Make an edge of a network to reach a target force.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The predicted edge force.
        """
        return jnp.abs(eq_state.forces[index]) * eq_state.lengths[index]
