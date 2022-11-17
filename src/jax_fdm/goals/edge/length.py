from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeLengthGoal(ScalarGoal, EdgeGoal):
    """
    Make an edge of a network to reach a certain length.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The current edge length.
        """
        return eq_state.lengths[index]
