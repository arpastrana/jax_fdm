from jax_fdm.constraints import Constraint


class NodeConstraint(Constraint):
    """
    Base class for all constraints that pertain to a node in a network.
    """
    def index_from_model(self, model):
        """
        The index of the node in a structure.
        """
        try:
            return model.structure.node_index[self.key]
        except TypeError:
            return tuple([model.structure.node_index[k] for k in self.key])
