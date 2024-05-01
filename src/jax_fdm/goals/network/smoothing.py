import jax.numpy as jnp

from jax import vmap

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
        self.adjacency = None
        self.indices_free = None

    def prediction(self, eq_state, index):
        """
        The current fairness value of the node.
        """
        xyz = eq_state.xyz[self.indices_free, :]
        adjacency = self.adjacency[self.indices_free, :]
        fairness_fn = vmap(node_nbrs_fairness_ngon, in_axes=(None, 0, 0))

        fairness_nodes = fairness_fn(eq_state.xyz, xyz, adjacency)
        fairness = jnp.sum(fairness_nodes)

        return jnp.atleast_1d(fairness)

    def init(self, model, structure):
        """
        Initialize the constraint with information from an equilibrium model.
        """
        super().init(model, structure)
        self.indices_free = structure.indices_free
        self.adjacency = structure.adjacency


def node_nbrs_fairness_ngon(xyz_all, xyz_node, adjacency_node):
    """
    Compute the fairness of an n-gon node neighborhood.
    """
    xyz_nbrs = adjacency_node @ xyz_all / jnp.sum(adjacency_node, axis=-1)

    fvector = xyz_node - xyz_nbrs
    assert fvector.shape == xyz_node.shape

    return jnp.sum(jnp.square(fvector))
