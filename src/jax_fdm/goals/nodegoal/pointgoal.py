from jax_fdm.goals import VectorGoal

from jax_fdm.goals.nodegoal import NodesGoal


class NodesPointGoal(VectorGoal, NodesGoal):
    """
    Make a node of a network to reach target xyz coordinates.
    """
    def __init__(self, keys, targets, weights=1.0):
        super().__init__(keys=keys, targets=targets, weights=weights)

    def prediction(self, eq_state):
        """
        The current xyz coordinates of the node in a network.
        """
        return eq_state.xyz[self.index, :]
