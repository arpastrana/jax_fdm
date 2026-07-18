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

    def __init__(self) -> None:
        super().__init__()
        # set in init() from the network structure, before any prediction runs
        self.adjacency: Float[Array, "nodes nodes"]

    def init(self, model: EquilibriumModel, structure: EquilibriumStructure) -> None:
        """
        Bind the goal to a structure, caching its adjacency matrix.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The structure whose adjacency matrix defines node neighborhoods.
        """
        super().init(model, structure)
        self.adjacency = structure.adjacency

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, "1"],
    ) -> Float[Array, ""]:
        """
        The mean fairness energy over the network's nodes.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read the node coordinates from.
        index :
            The sentinel index, unused.

        Returns
        -------
        prediction :
            The mean over nodes of the squared distance from each node to its
            neighbors' centroid.
        """
        xyz = eq_state.xyz
        adjacency = self.adjacency
        fairness_fn = vmap(node_nbrs_fairness_ngon, in_axes=(None, 0, 0))

        fairness_nodes = fairness_fn(xyz, xyz, adjacency)

        return jnp.mean(fairness_nodes)


def node_nbrs_fairness_ngon(
    xyz_all: Float[Array, "nodes 3"],
    xyz_node: Float[Array, "3"],
    adjacency_node: Float[Array, "nodes"],
) -> Float[Array, ""]:
    """
    Compute the fairness energy of one node's neighborhood.

    Parameters
    ----------
    xyz_all :
        The coordinates of all nodes.
    xyz_node :
        The coordinates of the node whose fairness is measured.
    adjacency_node :
        The row of the adjacency matrix selecting the node's neighbors.

    Returns
    -------
    fairness :
        The squared distance from the node to its neighbors' centroid.
    """
    num_nbrs = jnp.sum(adjacency_node, axis=-1)
    centroid = (adjacency_node @ xyz_all) / num_nbrs

    fvector = xyz_node - centroid
    assert fvector.shape == xyz_node.shape

    return jnp.dot(fvector, fvector)
