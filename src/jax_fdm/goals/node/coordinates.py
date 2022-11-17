from jax_fdm.goals import ScalarGoal

from jax_fdm.goals.node import NodeGoal


class NodeXCoordinateGoal(ScalarGoal, NodeGoal):
    """
    Make a node of a network to reach a target X coordinate.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The current X coordinate of the node in a network.
        """
        return eq_state.xyz[index, :1]


class NodeYCoordinateGoal(ScalarGoal, NodeGoal):
    """
    Make a node of a network to reach a target Y coordinate.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The current Y coordinate of the node in a network.
        """
        return eq_state.xyz[index, 1:2]


class NodeZCoordinateGoal(ScalarGoal, NodeGoal):
    """
    Make a node of a network to reach a target Z coordinate.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The current Z coordinate of the node in a network.
        """
        return eq_state.xyz[index, 2:]
