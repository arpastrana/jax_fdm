from jax_fdm.goals import Goal


class EdgeGoal(Goal):
    """
    Base class for all goals that pertain to an edge of a network.
    """
    def __init__(self, key, target, weight):
        super().__init__(key=key, target=target, weight=weight)

    def model_index(self, model):
        """
        The index of the goal key in a structure.
        """
        if isinstance(self.key, tuple) and len(self.key) == 2:
            return model.structure.edge_index[self.key]
        return tuple([model.structure.edge_index[key] for key in self.key])
