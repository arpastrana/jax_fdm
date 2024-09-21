from jax_fdm.constraints.vertex import VertexConstraint


class VertexXCoordinateConstraint(VertexConstraint):
    """
    Constraint the X coordinate of a vertex between a lower and an upper bound.
    """
    @staticmethod
    def constraint(eqstate, index):
        """
        Returns the X coordinate of a vertex from an equilibrium state.
        """
        return eqstate.xyz[index, :1]


class VertexYCoordinateConstraint(VertexConstraint):
    """
    Constraint the Y coordinate of a vertex between a lower and an upper bound.
    """
    @staticmethod
    def constraint(eqstate, index):
        """
        Returns the Y coordinate of a vertex from an equilibrium state.
        """
        return eqstate.xyz[index, 1:2]


class VertexZCoordinateConstraint(VertexConstraint):
    """
    Constraint the Z coordinate of a vertex between a lower and an upper bound.
    """
    @staticmethod
    def constraint(eqstate, index):
        """
        Returns the Z coordinate of a vertex from an equilibrium state.
        """
        return eqstate.xyz[index, 2]
