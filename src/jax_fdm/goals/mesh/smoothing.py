import jax.numpy as jnp
from jax import vmap
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumModel
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals import ScalarGoal
from jax_fdm.goals.mesh import MeshGoal


class MeshSmoothGoal(ScalarGoal, MeshGoal):
    """
    Smudge a mesh based on the smoothness energy of its vertices.

    Notes
    -----
    Smoothness, or fairness, is an energy that measures the squared distance
    between the position of a vertex and the centroid of its neighbors' positions.

    Based on Eq. 2 in Tang et al. (2013). DOI: 10.1145/2601097.2601213.
    The fairness is computed only for the free vertices, and each vertex term is
    reweighted by the square of its valence.
    """

    def __init__(self) -> None:
        super().__init__()
        # set in init() from the mesh structure, before any prediction runs
        self.adjacency: Float[Array, "vertices vertices"]
        self.indices_free: Int[Array, "nodes_free"]

    def init(
        self,
        model: EquilibriumModel,
        structure: EquilibriumMeshStructure,
    ) -> None:
        """
        Bind the goal to a mesh, caching its adjacency and free-vertex indices.

        Parameters
        ----------
        model :
            The equilibrium model.
        structure :
            The mesh structure whose adjacency and free-vertex indices are cached.
        """
        super().init(model, structure)
        self.adjacency = structure.adjacency
        self.indices_free = structure.indices_free

    def prediction(
        self,
        eq_state: EquilibriumState,
        index: Int[Array, "1"],
    ) -> Float[Array, ""]:
        """
        The mean fairness energy over the mesh's free vertices.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read vertex coordinates from.
        index :
            The sentinel index, unused.

        Returns
        -------
        prediction :
            The mean valence-weighted squared distance from each free vertex to its
            neighbors' centroid.
        """
        xyz = eq_state.xyz
        adjacency = self.adjacency
        fairness_fn = vmap(vertex_nbrs_fairness_ngon, in_axes=(None, 0, 0))

        fairness_vertices = fairness_fn(xyz, xyz, adjacency)
        fairness_vertices = fairness_vertices[self.indices_free]
        return jnp.mean(fairness_vertices)


def vertex_nbrs_fairness_ngon(
    xyz_all: Float[Array, "vertices 3"],
    xyz_vertex: Float[Array, "3"],
    adjacency_vertex: Float[Array, "vertices"],
) -> Float[Array, ""]:
    """
    Compute the valence-weighted fairness energy of one vertex's neighborhood.

    Parameters
    ----------
    xyz_all :
        The coordinates of all vertices.
    xyz_vertex :
        The coordinates of the vertex whose fairness is measured.
    adjacency_vertex :
        The row of the adjacency matrix selecting the vertex's neighbors.

    Returns
    -------
    fairness :
        The squared distance from the vertex to its neighbors' centroid, scaled by
        the square of its valence.
    """
    num_nbrs = jnp.sum(adjacency_vertex, axis=-1)
    centroid = (adjacency_vertex @ xyz_all) / num_nbrs

    fvector = xyz_vertex - centroid
    assert fvector.shape == xyz_vertex.shape

    # return jnp.dot(fvector, fvector)
    return jnp.dot(fvector, fvector) * jnp.square(num_nbrs)
