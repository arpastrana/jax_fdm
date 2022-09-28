from jax_fdm.goals import Goal


class NodeGoal(Goal):
    """
    Base class for all goals that pertain to the node of a network.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    def index_from_model(self, model):
        """
        The index of the edge in a structure.
        """
        try:
            return model.structure.node_index[self.key]
        except TypeError:
            return tuple([model.structure.node_index[k] for k in self.key])
