from jax_fdm.goals import Goal


class EdgeGoal(Goal):
    """
    Base class for all goals that pertain to an edge of a network.
    """
    def __init__(self, key, target, weight=1.0):
        super().__init__(key=key, target=target, weight=weight)

    @staticmethod
    def index_from_model(model, key):
        """
        The index of the edge key in an equilibrium structure.
        """
        try:
            return model.structure.edge_index[key]
        except TypeError:
            return tuple([model.structure.edge_index[k] for k in key])
