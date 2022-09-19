from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edgegoal import EdgesGoal


class EdgesLengthGoal(ScalarGoal, EdgesGoal):
    """
    Make an edge of a network to reach a certain length.
    """
    def __init__(self, keys, targets, weights=1.0):
        super().__init__(keys=keys, targets=targets, weights=weights)

    def prediction(self, eq_state):
        """
        The current edge length.
        """
        return eq_state.lengths[self.index, ]
