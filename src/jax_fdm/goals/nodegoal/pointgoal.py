import jax.numpy as jnp

from jax_fdm.goals import ScalarGoal
from jax_fdm.goals import VectorGoal

from jax_fdm.goals.nodegoal import NodeGoal


class NodePointGoal(VectorGoal, NodeGoal):
    """
    Make a node of a network to reach target xyz coordinates.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The current xyz coordinates of the node in a network.
        """
        return eq_state.xyz[index, :]


class NodePointZGoal(ScalarGoal, NodeGoal):
    """
    Make a node of a network to reach target xyz coordinates.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The current xyz coordinates of the node in a network.
        """
        # return jnp.reshape(eq_state.xyz[index, 2], (-1, 1))
        return eq_state.xyz[index, 2:]
