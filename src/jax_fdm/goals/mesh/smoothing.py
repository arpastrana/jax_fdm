import jax.numpy as jnp

from jax import vmap

from jax_fdm.goals import ScalarGoal

from jax_fdm.goals.mesh import MeshGoal


class MeshSmoothGoal(ScalarGoal, MeshGoal):
    """
    Smudge a mesh based on the smoothness energy of its vertices.

    Notes
    -----
    Smoothness, or fairness, is an energy that measures the squared distance
    between the position of a vertex and the centroid of its neighbors' positions.

    Based on Eq. 2 in Tang et al. (2013). DOI: 10.1145/2601097.2601213
    Modifications:
        - The fairness is computed only for the free vertices.
        - No reweighting is performed based on vertex valences.
    """
    def __init__(self):
        super().__init__()
        self.adjacency = None
        self.indices_free = None

    def init(self, model, structure):
        """
        Initialize the constraint with information from an equilibrium model.
        """
        super().init(model, structure)
        self.adjacency = structure.adjacency
        self.indices_free = structure.indices_free

    def prediction(self, eq_state, index):
        """
        The current smoothness of the vertex.
        """
        xyz = eq_state.xyz
        adjacency = self.adjacency
        fairness_fn = vmap(vertex_nbrs_fairness_ngon, in_axes=(None, 0, 0))

        fairness_vertices = fairness_fn(xyz, xyz, adjacency)
        fairness_vertices = fairness_vertices[self.indices_free]
        fairness = jnp.mean(fairness_vertices)

        return jnp.atleast_1d(fairness)


def vertex_nbrs_fairness_ngon(xyz_all, xyz_vertex, adjacency_vertex):
    """
    Compute the fairness of an n-gon vertex neighborhood.
    """
    num_nbrs = jnp.sum(adjacency_vertex, axis=-1)
    centroid = (adjacency_vertex @ xyz_all) / num_nbrs

    fvector = xyz_vertex - centroid
    assert fvector.shape == xyz_vertex.shape

    return jnp.dot(fvector, fvector) # * jnp.square(num_nbrs)
