from jax_fdm.constraints import Constraint


class NetworkConstraint(Constraint):
    """
    Base class for all constraints that pertain to all the edges or all the nodes of a network.
    """
    def __init__(self, bound_low, bound_up):
        super().__init__(key=-1, bound_low=bound_low, bound_up=bound_up)

    def index_from_model(self, model):
        """
        Method only for API compatibility.
        """
        return -1
