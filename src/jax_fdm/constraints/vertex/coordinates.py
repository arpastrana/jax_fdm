from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.vertex import VertexConstraint
from jax_fdm.equilibrium import EquilibriumState


class VertexXCoordinateConstraint(VertexConstraint):
    """
    Bound the X coordinate of a vertex between a lower and an upper value.
    """

    def constraint(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The X coordinate of the vertex.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinate from.
        index :
            The index of the vertex.

        Returns
        -------
        constraint :
            The vertex's X coordinate.
        """
        return eq_state.xyz[index, 0]


class VertexYCoordinateConstraint(VertexConstraint):
    """
    Bound the Y coordinate of a vertex between a lower and an upper value.
    """

    def constraint(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The Y coordinate of the vertex.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinate from.
        index :
            The index of the vertex.

        Returns
        -------
        constraint :
            The vertex's Y coordinate.
        """
        return eq_state.xyz[index, 1]


class VertexZCoordinateConstraint(VertexConstraint):
    """
    Bound the Z coordinate of a vertex between a lower and an upper value.
    """

    def constraint(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The Z coordinate of the vertex.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinate from.
        index :
            The index of the vertex.

        Returns
        -------
        constraint :
            The vertex's Z coordinate.
        """
        return eq_state.xyz[index, 2]
