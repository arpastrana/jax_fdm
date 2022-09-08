from dfdm.goals import Goal


class NodeGoal(Goal):
    """
    Base class for all goals that pertain to a nodes of a network.
    """
    def __init__(self, key, target, weight):
        super().__init__(key=key, target=target, weight=weight)

    def index(self, model):
        """
        The index of the goal key in a structure.
        """
        return model.structure.node_index[self.key()]

    def key(self):
        """
        The key of the node in the network.
        """
        return self._key
