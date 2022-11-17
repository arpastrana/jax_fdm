from jax_fdm.constraints.node import NodeConstraint


class NodeXCoordinateConstraint(NodeConstraint):
    """
    Constraint the X coordinate of a node between a lower and an upper bound.
    """
    @staticmethod
    def constraint(eqstate, index):
        """
        Returns the X coordinate of a node from an equilibrium state.
        """
        return eqstate.xyz[index, :1]


class NodeYCoordinateConstraint(NodeConstraint):
    """
    Constraint the Y coordinate of a node between a lower and an upper bound.
    """
    @staticmethod
    def constraint(eqstate, index):
        """
        Returns the Y coordinate of a node from an equilibrium state.
        """
        return eqstate.xyz[index, 1:2]


class NodeZCoordinateConstraint(NodeConstraint):
    """
    Constraint the Z coordinate of a node between a lower and an upper bound.
    """
    @staticmethod
    def constraint(eqstate, index):
        """
        Returns the Z coordinate of a node from an equilibrium state.
        """
        return eqstate.xyz[index, 2]
