from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals import VectorGoal
from jax_fdm.goals.node import NodeGoal


class NodePointGoal(VectorGoal, NodeGoal):
    """
    Make a node of a network to reach target xyz coordinates.
    """
    @staticmethod
    def prediction(eq_state: EquilibriumState, index: Int[Array, ""]) -> Float[Array, "3"]:
        """
        The current xyz coordinates of the node in a network.
        """
        return eq_state.xyz[index, :]
