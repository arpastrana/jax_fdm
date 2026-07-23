from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals.node.node import NodeGoal

__all__ = ["NodePointGoal"]


class NodePointGoal(NodeGoal):
    """
    Drive a node toward target xyz coordinates.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumStructure,
        index: Int[Array, ""],
    ) -> Float[Array, "3"]:
        """
        The current xyz coordinates of the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinates from.
        structure :
            The structure the goal is evaluated against; unused.
        index :
            The index of the node.

        Returns
        -------
        prediction :
            The node's xyz coordinates.
        """
        return eq_state.xyz[index, :]
