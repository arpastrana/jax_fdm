from jax_fdm.goals import Goal


class NetworkGoal(Goal):
    """
    Base class for all goals that pertain to all the nodes and edgees of a network.
    """
    def __init__(self, keys, target, weight):
        super().__init__(key=keys, target=target, weight=weight)

    def model_index(self, model):
        """
        The index of the goal key in a structure.
        """
        pass
