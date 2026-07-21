from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.constraints.node.node import NodeConstraint
from jax_fdm.equilibrium import EquilibriumState


class NodeXCoordinateConstraint(NodeConstraint):
    """
    Bound the X coordinate of a node between a lower and an upper value.
    """

    def constraint(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The X coordinate of the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinate from.
        index :
            The index of the node.

        Returns
        -------
        constraint :
            The node's X coordinate.
        """
        return eq_state.xyz[index, 0]


class NodeYCoordinateConstraint(NodeConstraint):
    """
    Bound the Y coordinate of a node between a lower and an upper value.
    """

    def constraint(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The Y coordinate of the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinate from.
        index :
            The index of the node.

        Returns
        -------
        constraint :
            The node's Y coordinate.
        """
        return eq_state.xyz[index, 1]


class NodeZCoordinateConstraint(NodeConstraint):
    """
    Bound the Z coordinate of a node between a lower and an upper value.
    """

    def constraint(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The Z coordinate of the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinate from.
        index :
            The index of the node.

        Returns
        -------
        constraint :
            The node's Z coordinate.
        """
        return eq_state.xyz[index, 2]
