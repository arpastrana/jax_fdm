import jax.numpy as jnp
from jax.experimental.sparse import BCOO
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
        self.adjacency: Float[Array, "nodes nodes"] | Float[BCOO, "nodes nodes"]

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
        fairness_nodes = nodes_nbrs_fairness(xyz, self.adjacency)

        return jnp.mean(fairness_nodes)


def nodes_nbrs_fairness(
    xyz: Float[Array, "nodes 3"],
    adjacency: Float[Array, "nodes nodes"] | Float[BCOO, "nodes nodes"],
) -> Float[Array, "nodes"]:
    """
    Compute the fairness energy of every node's neighborhood.

    Parameters
    ----------
    xyz :
        The coordinates of all nodes.
    adjacency :
        The adjacency matrix selecting each node's neighbors.

    Returns
    -------
    fairness :
        The squared distance from each node to its neighbors' centroid.

    Notes
    -----
    Formulated as adjacency matrix products so that the matrix can be stored
    dense or in sparse format. A node without neighbors has a nan energy.
    """
    num_nbrs = adjacency @ jnp.ones(xyz.shape[0])
    centroids = (adjacency @ xyz) / num_nbrs[:, None]

    fvectors = xyz - centroids

    return jnp.sum(jnp.square(fvectors), axis=-1)
