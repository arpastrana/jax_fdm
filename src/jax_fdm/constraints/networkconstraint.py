from jax_fdm.constraints import Constraint


class NetworkConstraint(Constraint):
    pass


class NetworkEdgesLengthConstraint(NetworkConstraint):
    """
    Set constraint bounds to the length of all the edges of a network.
    """
    def constraint(self, eqstate, *args, **kwargs):
        """
        The constraint function relative to a equilibrium state.
        """
        return eqstate.lengths


class NetworkEdgesForceConstraint(NetworkConstraint):
    """
    Set constraint bounds to the length of all the edges of a network.
    """
    def constraint(self, eqstate, *args, **kwargs):
        """
        The constraint function relative to a equilibrium state.
        """
        return eqstate.forces
