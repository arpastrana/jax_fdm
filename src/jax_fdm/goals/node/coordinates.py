from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.goal import ScalarGoal
from jax_fdm.goals.node.node import NodeGoal

__all__ = [
    "NodeXCoordinateGoal",
    "NodeYCoordinateGoal",
    "NodeZCoordinateGoal",
]


class NodeXCoordinateGoal(ScalarGoal, NodeGoal):
    """
    Drive a node toward a target X coordinate.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumStructure,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        The current X coordinate of the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinate from.
        structure :
            The structure the goal is evaluated against; unused.
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
        structure: EquilibriumStructure,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        The current Y coordinate of the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinate from.
        structure :
            The structure the goal is evaluated against; unused.
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
        structure: EquilibriumStructure,
        index: Int[Array, ""],
    ) -> Float[Array, "1"]:
        """
        The current Z coordinate of the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinate from.
        structure :
            The structure the goal is evaluated against; unused.
        index :
            The index of the node.

        Returns
        -------
        prediction :
            The node's Z coordinate.
        """
        return eq_state.xyz[index, 2:]
