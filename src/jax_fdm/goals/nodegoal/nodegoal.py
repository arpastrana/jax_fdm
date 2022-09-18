from jax_fdm.goals import Goal


class NodesGoal(Goal):
    """
    Base class for all goals that pertain to a selection of the nodes of a network.
    """
    def __init__(self, keys, targets, weights):
        super().__init__(key=kesy, target=targets, weight=weights)

    def model_index(self, model):
        """
        The index of the goal key in a structure.
        """
        if isinstance(self.key, int):
            return model.structure.node_index[self.key]
        return tuple([model.structure.node_index[key] for key in self.key])
