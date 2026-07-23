import jax.numpy as jnp
from jaxtyping import Array
from jaxtyping import Float
from jaxtyping import Int

from jax_fdm.equilibrium import EquilibriumMeshStructure
from jax_fdm.equilibrium import EquilibriumState
from jax_fdm.goals.goal import TargetLike
from jax_fdm.goals.mesh.mesh import MeshGoal
from jax_fdm.goals.network.laplacian import NetworkXYZLaplacianGoal

__all__ = ["MeshXYZLaplacianGoal", "MeshXYZFaceLaplacianGoal"]


class MeshXYZLaplacianGoal(NetworkXYZLaplacianGoal):
    """
    Minimize the Laplacian energy of a mesh's coordinates.

    Notes
    -----
    A thin wrapper of
    [NetworkXYZLaplacianGoal][jax_fdm.goals.network.laplacian.NetworkXYZLaplacianGoal]
    that fixes the key to the mesh sentinel.
    """

    def __init__(
        self,
        target: TargetLike = 0.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(target=target, weight=weight)


class MeshXYZFaceLaplacianGoal(MeshGoal):
    """
    Minimize the Laplacian energy of the XYZ faces coordinates of a mesh.

    Notes
    -----
    This goal can be handy to "smoothen" the looks of a mesh.

    This energy is computed as the distance of the XYZ coordinates of
    every vertex to the centroid of the centroids of the faces the
    vertex is part of.

    An energy-minimizing mesh will have every vertex as close as possible
    to its neighboring faces centroid.
    """

    def laplacian_vertices(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumMeshStructure,
    ) -> Float[Array, "vertices"]:
        """
        The per-vertex Laplacian energy against neighboring face centroids.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read vertex coordinates from.
        structure :
            The mesh structure providing the face-vertex connectivity.

        Returns
        -------
        laplacians :
            The squared distance from each vertex to the mean of its incident face
            centroids.

        Notes
        -----
        Formulated as products with the face-vertex connectivity so that the
        matrix can be stored dense or in sparse format. Each vertex's neighbor
        centroid is the average of its incident face centroids, weighted by the
        row-normalized connectivity entries.
        """
        xyz = eq_state.xyz
        connectivity = structure.connectivity_faces_vertices

        faces_centroid = connectivity @ xyz

        # Use .transpose() rather than the .T property: on the sparse union the
        # untyped .T property infers as None, while .transpose() is typed to
        # return the matrix. They are identical for this rank-2 incidence matrix.
        connectivity_t = connectivity.transpose()
        weights = connectivity_t @ jnp.ones(connectivity.shape[0])
        nbrs_centroid = (connectivity_t @ faces_centroid) / weights[:, None]

        return jnp.sum(jnp.square(xyz - nbrs_centroid), axis=-1)

    def prediction(
        self,
        eq_state: EquilibriumState,
        structure: EquilibriumMeshStructure,
        index: Int[Array, ""],
    ) -> Float[Array, ""]:
        """
        The mean per-vertex Laplacian energy of the mesh.

        Parameters
        ----------
        eq_state :
            The equilibrium state to read vertex coordinates from.
        structure :
            The mesh structure providing the face-vertex connectivity.
        index :
            The sentinel index, unused.

        Returns
        -------
        prediction :
            The mean over vertices of the face-centroid Laplacian energy.
        """
        laplacians = self.laplacian_vertices(eq_state, structure)

        return jnp.mean(laplacians)
