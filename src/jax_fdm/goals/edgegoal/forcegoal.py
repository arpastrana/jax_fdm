from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edgegoal import EdgesGoal


class EdgesForceGoal(ScalarGoal, EdgesGoal):
    """
    Make an edge of a network to reach a certain force.
    """
    def __init__(self, keys, targets, weights=1.0):
        super().__init__(key=keys, target=targets, weight=weights)

    def prediction(self, eq_state):
        """
        The current edge length.
        """
        return eq_state.forces[self.index, ]
