from jax_fdm.goals import Goal


class EdgesGoal(Goal):
    """
    Base class for all goals that pertain to an edge of a network.
    """
    def __init__(self, keys, targets, weights):
        super().__init__(key=keys, target=targets, weight=weights)

    def model_index(self, model):
        """
        The index of the goal key in a structure.
        """
        return tuple([model.structure.edge_index[key] for key in self.key])
