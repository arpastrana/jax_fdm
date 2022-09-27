from jax_fdm.constraints import Constraint


class NodeConstraint(Constraint):
    """
    Base class for all constraints that pertain to a node in a network.
    """
    @staticmethod
    def index_from_model(model, key):
        """
        The index of the node in a structure.
        """
        try:
            return model.structure.node_index[key]
        except TypeError:
            return tuple([model.structure.node_index[k] for k in key])
