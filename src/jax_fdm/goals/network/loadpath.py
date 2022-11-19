import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal

from jax_fdm.goals.network import NetworkGoal


class NetworkLoadPathGoal(ScalarGoal, NetworkGoal):
    """
    Make the total load path of a network to reach a target magnitude.

    The load path of an edge is the absolute value of the product of the
    the force on the edge time its length.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The current load path of the network.
        """
        load_path = jnp.sum(jnp.abs(jnp.multiply(eq_state.lengths, eq_state.forces)))

        return jnp.atleast_1d(load_path)
