from jax_fdm.goals import Goal


class NetworkGoal(Goal):
    """
    Base class for all goals that pertain to all the nodes and edgees of a network.
    """
    def __init__(self, key, target, weight):
        super().__init__(key=key, target=target, weight=weight)

    def index(self, model):
        """
        This method exists to comply with thee current APU but it returns `None`.
        """
        return

    def key(self):
        """
        This method exists to comply with thee current APU but it returns `None`.
        """
        return
