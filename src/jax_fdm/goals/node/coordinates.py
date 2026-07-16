from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.node import NodeGoal


class NodeXCoordinateGoal(ScalarGoal, NodeGoal):
    """
    Drive a node toward a target X coordinate.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        The current X coordinate of the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinate from.
        index :
            The index of the node.

        Returns
        -------
        prediction :
            The node's X coordinate.
        """
        return eq_state.xyz[index, :1]


class NodeYCoordinateGoal(ScalarGoal, NodeGoal):
    """
    Drive a node toward a target Y coordinate.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        The current Y coordinate of the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinate from.
        index :
            The index of the node.

        Returns
        -------
        prediction :
            The node's Y coordinate.
        """
        return eq_state.xyz[index, 1:2]


class NodeZCoordinateGoal(ScalarGoal, NodeGoal):
    """
    Drive a node toward a target Z coordinate.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        The current Z coordinate of the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinate from.
        index :
            The index of the node.

        Returns
        -------
        prediction :
            The node's Z coordinate.
        """
        return eq_state.xyz[index, 2:]
