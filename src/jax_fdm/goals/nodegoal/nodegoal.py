from jax_fdm.goals import Goal


class NodeGoal(Goal):
    """
    Base class for all goals that pertain to a selection of the nodes of a network.
    """
    def __init__(self, key, target, weight):
        super().__init__(key=key, target=target, weight=weight)

    def model_index(self, model):
        """
        The index of the goal key in a structure.
        """
        if isinstance(self.key, int):
            return model.structure.node_index[self.key]
        return tuple([model.structure.node_index[key] for key in self.key])
