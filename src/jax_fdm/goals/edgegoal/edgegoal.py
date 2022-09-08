from dfdm.goals import Goal


class EdgeGoal(Goal):
    """
    Base class for all goals that pertain to an edge of a network.
    """
    def __init__(self, key, target, weight):
        super().__init__(key=key, target=target, weight=weight)

    def index(self, model):
        """
        The index of the goal key in a structure.
        """
        return model.structure.edge_index[self.key()]

    def key(self):
        """
        The key of the node in the network.
        """
        return self._key
