import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals.mesh.mesh import MeshGoal

__all__ = [
    "MeshSmoothGoal",
    "vertices_nbrs_fairness",
]


class MeshSmoothGoal(MeshGoal):
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

    def prediction(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumMeshStructure,
        index: Int[Array, "1"],
    ) -> Float[Array, ""]:
        """
        The mean fairness energy over the mesh's free vertices.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read vertex coordinates from.
        structure :
            The mesh structure providing the adjacency and free-vertex indices.
        index :
            The sentinel index, unused.

        Returns
        -------
        prediction :
            The mean valence-weighted squared distance from each free vertex to its
            neighbors' centroid.
        """
        xyz = eq_state.xyz
        fairness_vertices = vertices_nbrs_fairness(xyz, structure.adjacency)
        fairness_vertices = fairness_vertices[structure.indices_free]

        return jnp.mean(fairness_vertices)


def vertices_nbrs_fairness(
    xyz: Float[Array, "vertices 3"],
    adjacency: Float[Array, "vertices vertices"] | Float[BCOO, "vertices vertices"],
) -> Float[Array, "vertices"]:
    """
    Compute the valence-weighted fairness energy of every vertex's neighborhood.

    Parameters
    ----------
    xyz :
        The coordinates of all vertices.
    adjacency :
        The adjacency matrix selecting each vertex's neighbors.

    Returns
    -------
    fairness :
        The squared distance from each vertex to its neighbors' centroid, scaled
        by the square of its valence.

    Notes
    -----
    Formulated as adjacency matrix products so that the matrix can be stored
    dense or in sparse format. A vertex without neighbors has a nan energy.
    """
    num_nbrs = adjacency @ jnp.ones(xyz.shape[0])
    centroids = (adjacency @ xyz) / num_nbrs[:, None]

    fvectors = xyz - centroids

    return jnp.sum(jnp.square(fvectors), axis=-1) * jnp.square(num_nbrs)
