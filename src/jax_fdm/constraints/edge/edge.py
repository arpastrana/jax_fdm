from jax_fdm.constraints import Constraint


class EdgeConstraint(Constraint):
    """
    Base class for all constraints that pertain to an edge of a network.
    """
    def index_from_model(self, model, structure):
        """
        The index of the edge key in an equilibrium structure.
        """
        try:
            return structure.edge_index[self.key]
        except TypeError:
            return tuple([structure.edge_index[k] for k in self.key])
