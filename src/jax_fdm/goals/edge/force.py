from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.edge import EdgeGoal


class EdgeForceGoal(ScalarGoal, EdgeGoal):
    """
    Make an edge of a network to reach a target force.
    """
    @staticmethod
    def prediction(eq_state, index):
        """
        The predicted edge force.
        """
        return eq_state.forces[index]
