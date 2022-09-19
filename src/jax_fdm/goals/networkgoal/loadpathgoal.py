import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.networkgoal import NetworkGoal


class NetworkLoadPathGoal(ScalarGoal, NetworkGoal):
    """
    Make the total load path of a network to reach a target magnitude.

    The load path of an edge is the absolute value of the product of the
    the force on the edge time its length.
    """
    def __init__(self, target=None, weight=1.0):
        super().__init__(targets=target, weight=weight)

    def prediction(self, eq_state, *args, **kwargs):
        """
        The current load path of the network.
        """
        load_path = jnp.sum(jnp.abs(eq_state.lengths * eq_state.forces))

        return jnp.atleast_1d(load_path)
