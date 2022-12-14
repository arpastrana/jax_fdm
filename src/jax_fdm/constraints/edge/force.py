from jax_fdm.constraints.edge import EdgeConstraint


class EdgeForceConstraint(EdgeConstraint):
    """
    Constraints the force of an edge between a lower and an upper bound.
    """
    @staticmethod
    def constraint(eqstate, index):
        """
        Returns the length of an edge from an equilibrium state.
        """
        return eqstate.forces[index, :]
