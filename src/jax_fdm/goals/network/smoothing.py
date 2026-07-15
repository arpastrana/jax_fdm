import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.equilibrium import EquilibriumStructure
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.network import NetworkGoal


class NetworkSmoothGoal(ScalarGoal, NetworkGoal):
    """
    Smudge a network based on the fairness of its nodes.

    Notes
    -----
    Fairness is an energy that measures the smoothness of the
    position of a node w.r.t. to that of its neighbors.

    The fairness energy is computed as the squared length of the
    vector between every node position and their neighbors' centroid.
    """
    def __init__(self):
        super().__init__()
        # set in init() from the network structure, before any prediction runs
        self.adjacency: Float[Array, "nodes nodes"]

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Initialize the constraint with information from an equilibrium model.
        """
        super().init(model, structure)
        self.adjacency = structure.adjacency

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, ""],
        ) -> Float[Array, "1"]:
        """
        The current fairness value of the node.
        """
        xyz = eq_state.xyz
        adjacency = self.adjacency
        fairness_fn = vmap(node_nbrs_fairness_ngon, in_axes=(None, 0, 0))

        fairness_nodes = fairness_fn(xyz, xyz, adjacency)
        fairness = jnp.mean(fairness_nodes)

        return jnp.atleast_1d(fairness)


def node_nbrs_fairness_ngon(
    xyz_all: Float[Array, "nodes 3"],
    xyz_node: Float[Array, "3"],
    adjacency_node: Float[Array, "nodes"],
) -> Float[Array, ""]:
    """
    Compute the fairness of an n-gon node neighborhood.
    """
    num_nbrs = jnp.sum(adjacency_node, axis=-1)
    centroid = (adjacency_node @ xyz_all) / num_nbrs

    fvector = xyz_node - centroid
    assert fvector.shape == xyz_node.shape

    return jnp.dot(fvector, fvector)
