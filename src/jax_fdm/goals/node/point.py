from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals.goal import VectorGoal
from jax_fdm.goals.node.node import NodeGoal


class NodePointGoal(VectorGoal, NodeGoal):
    """
    Drive a node toward target xyz coordinates.
    """

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
    ) -> Float[Array, "3"]:
        """
        The current xyz coordinates of the node.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the coordinates from.
        index :
            The index of the node.

        Returns
        -------
        prediction :
            The node's xyz coordinates.
        """
        return eq_state.xyz[index, :]
